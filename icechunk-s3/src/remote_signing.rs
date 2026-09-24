//! Remote request signing, following the Iceberg REST catalog S3 signer protocol.
//!
//! With remote signing the client holds no S3 credentials. The AWS SDK is configured
//! without credentials (so it sends unsigned requests), and the HTTP connector is
//! wrapped: right before each request attempt is transmitted, the wrapper asks the
//! remote signer (e.g. Lakekeeper) to sign it, applies the returned headers/URI, and
//! then hands the request to the real connector.
//!
//! This is done at the connector level because it's the only async hook in the SDK
//! request pipeline; interceptors and signers are synchronous. Since retries call the
//! connector again, every attempt gets a fresh signature.
//!
//! Protocol (see `s3-signer-open-api.yaml` in the Iceberg repository):
//!
//! ```text
//! POST <signer_url>
//! Authorization: Bearer <token>
//! {"region": "...", "uri": "...", "method": "PUT", "headers": {"name": ["value"]}, "body": "..."}
//!
//! 200 OK
//! {"uri": "...", "headers": {"Authorization": ["AWS4-HMAC-SHA256 ..."], "x-amz-date": ["..."]}}
//! ```

use std::{collections::HashMap, sync::Arc};

use aws_sdk_s3::primitives::ByteStream;
use aws_smithy_runtime_api::{
    client::{
        http::{
            HttpClient, HttpConnector, HttpConnectorFuture, HttpConnectorSettings,
            SharedHttpClient, SharedHttpConnector,
        },
        orchestrator::HttpRequest,
        result::ConnectorError,
        runtime_components::{RuntimeComponents, RuntimeComponentsBuilder},
        runtime_plugin::RuntimePlugin as _,
    },
    http::Request,
};
use aws_smithy_types::{body::SdkBody, retry::ErrorKind};
use bytes::Bytes;
use chrono::{TimeDelta, Utc};
use icechunk_storage::s3_config::{S3RemoteSigningConfig, S3SignerToken};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

/// The SDK doesn't compute a payload hash for unsigned (anonymous) requests, but
/// `SigV4` needs one. We don't send the body to the signer, so we use the S3
/// `UNSIGNED-PAYLOAD` convention, like the Iceberg clients do.
const CONTENT_SHA256_HEADER: &str = "x-amz-content-sha256";
const UNSIGNED_PAYLOAD: &str = "UNSIGNED-PAYLOAD";

#[derive(Debug, Serialize)]
struct SignRequest<'a> {
    region: &'a str,
    uri: &'a str,
    method: &'a str,
    headers: HashMap<String, Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    body: Option<&'a str>,
}

#[derive(Debug, Deserialize)]
struct SignResponse {
    uri: String,
    #[serde(default)]
    headers: HashMap<String, Vec<String>>,
}

/// The HTTP client the SDK would use by default.
pub(crate) fn default_http_client() -> Option<SharedHttpClient> {
    let plugin = aws_smithy_runtime::client::defaults::default_http_client_plugin_v2(
        aws_config::BehaviorVersion::v2026_01_12(),
    )?;
    plugin
        .runtime_components(&RuntimeComponentsBuilder::new("icechunk_remote_signing"))
        .http_client()
}

/// How long before expiration a fetched token is considered stale.
const TOKEN_EXPIRY_BUFFER: TimeDelta = TimeDelta::seconds(60);

/// State shared by all connectors of one S3 client.
#[derive(Debug)]
struct Signer {
    config: S3RemoteSigningConfig,
    region: String,
    /// Last token returned by `config.token_fetcher`. The async mutex makes
    /// concurrent requests wait for a single refresh.
    cached_token: Mutex<Option<S3SignerToken>>,
}

impl Signer {
    /// The bearer token to use, fetching a new one if needed. `stale` is a token
    /// the signer just rejected.
    async fn token(&self, stale: Option<&str>) -> Result<Option<String>, ConnectorError> {
        let Some(fetcher) = &self.config.token_fetcher else {
            return Ok(self.config.token.clone());
        };
        let mut cached = self.cached_token.lock().await;
        let usable = cached.as_ref().is_some_and(|t| {
            Some(t.token.as_str()) != stale
                && t.expires_after
                    .is_none_or(|exp| exp > Utc::now() + TOKEN_EXPIRY_BUFFER)
        });
        if !usable {
            let token = fetcher.get().await.map_err(|e| {
                ConnectorError::other(
                    format!("cannot fetch remote signer token: {e}").into(),
                    None,
                )
            })?;
            *cached = Some(token);
        }
        Ok(cached.as_ref().map(|t| t.token.clone()))
    }
}

/// Wraps the SDK's default [`HttpClient`] so that every request is remotely signed.
#[derive(Debug)]
pub(crate) struct RemoteSigningHttpClient {
    inner: SharedHttpClient,
    signer: Arc<Signer>,
}

impl RemoteSigningHttpClient {
    pub(crate) fn new(
        inner: SharedHttpClient,
        config: S3RemoteSigningConfig,
        region: String,
    ) -> Self {
        let signer = Signer { config, region, cached_token: Mutex::new(None) };
        Self { inner, signer: Arc::new(signer) }
    }
}

impl HttpClient for RemoteSigningHttpClient {
    fn http_connector(
        &self,
        settings: &HttpConnectorSettings,
        components: &RuntimeComponents,
    ) -> SharedHttpConnector {
        SharedHttpConnector::new(RemoteSigningConnector {
            inner: self.inner.http_connector(settings, components),
            signer: Arc::clone(&self.signer),
        })
    }

    fn connector_metadata(
        &self,
    ) -> Option<aws_smithy_runtime_api::client::connector_metadata::ConnectorMetadata>
    {
        self.inner.connector_metadata()
    }
}

#[derive(Debug)]
struct RemoteSigningConnector {
    inner: SharedHttpConnector,
    signer: Arc<Signer>,
}

impl HttpConnector for RemoteSigningConnector {
    fn call(&self, mut request: HttpRequest) -> HttpConnectorFuture {
        let inner = self.inner.clone();
        let signer = Arc::clone(&self.signer);
        HttpConnectorFuture::new(async move {
            sign_request(&inner, &signer, &mut request).await?;
            inner.call(request).await
        })
    }
}

async fn sign_request(
    connector: &SharedHttpConnector,
    signer: &Signer,
    request: &mut HttpRequest,
) -> Result<(), ConnectorError> {
    if !request.headers().contains_key(CONTENT_SHA256_HEADER) {
        request.headers_mut().insert(CONTENT_SHA256_HEADER, UNSIGNED_PAYLOAD);
    }

    let mut headers: HashMap<String, Vec<String>> = HashMap::new();
    for (name, value) in request.headers() {
        headers.entry(name.to_string()).or_default().push(value.to_string());
    }

    // Per the protocol, the body is only sent for requests whose semantics are not
    // fully described by the URI, e.g. `DeleteObjects` (POST ?delete), so the signer
    // can authorize every key being deleted.
    let body = if request.method() == "POST" {
        request.body().bytes().and_then(|b| std::str::from_utf8(b).ok())
    } else {
        None
    };

    let payload = Bytes::from(
        serde_json::to_vec(&SignRequest {
            region: &signer.region,
            uri: request.uri(),
            method: request.method(),
            headers,
            body,
        })
        .map_err(|e| ConnectorError::other(e.into(), None))?,
    );

    let token = signer.token(None).await?;
    let (mut status, mut body) =
        call_signer(connector, &signer.config, token.as_deref(), payload.clone()).await?;
    // A rejected token that came from a fetcher may have been revoked or expired
    // early: fetch a new one and try once more.
    if matches!(status, 401 | 403) && signer.config.token_fetcher.is_some() {
        let fresh = signer.token(token.as_deref()).await?;
        (status, body) =
            call_signer(connector, &signer.config, fresh.as_deref(), payload).await?;
    }

    if !(200..300).contains(&status) {
        let kind = (status >= 500 || status == 429).then_some(ErrorKind::TransientError);
        let msg = format!(
            "remote signer returned HTTP {status}: {}",
            String::from_utf8_lossy(&body)
        );
        return Err(ConnectorError::other(msg.into(), kind));
    }

    let signed: SignResponse = serde_json::from_slice(&body).map_err(|e| {
        ConnectorError::other(format!("invalid remote signer response: {e}").into(), None)
    })?;

    if signed.uri != request.uri() {
        request.set_uri(signed.uri).map_err(|e| ConnectorError::other(e.into(), None))?;
    }
    let out_headers = request.headers_mut();
    for (name, values) in signed.headers {
        out_headers.remove(&name);
        for value in values {
            out_headers.append(name.clone(), value);
        }
    }
    Ok(())
}

/// Sends one signing request, returning the HTTP status and response body.
async fn call_signer(
    connector: &SharedHttpConnector,
    config: &S3RemoteSigningConfig,
    token: Option<&str>,
    payload: Bytes,
) -> Result<(u16, Bytes), ConnectorError> {
    let mut sign_req = Request::new(SdkBody::from(payload));
    sign_req.set_method("POST").map_err(|e| ConnectorError::other(e.into(), None))?;
    sign_req
        .set_uri(config.signer_url.as_str())
        .map_err(|e| ConnectorError::other(e.into(), None))?;
    let sign_headers = sign_req.headers_mut();
    sign_headers.insert("content-type", "application/json");
    sign_headers.insert("accept", "application/json");
    if let Some(token) = token {
        sign_headers.insert("authorization", format!("Bearer {token}"));
    }
    for (name, value) in &config.headers {
        sign_headers.insert(name.clone(), value.clone());
    }

    let response = connector.call(sign_req).await?;
    let status = response.status().as_u16();
    let body = ByteStream::new(response.into_body())
        .collect()
        .await
        .map_err(|e| ConnectorError::io(e.into()))?
        .into_bytes();
    Ok((status, body))
}
