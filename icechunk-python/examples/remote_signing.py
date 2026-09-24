"""Remote request signing: the client holds no S3 credentials.

Every request to the object store is first sent (method, URI, headers) to a signer
service, which returns SigV4 headers. This is the Iceberg REST catalog "S3 remote
signing" protocol, which Lakekeeper implements.

This script starts a tiny mock signer (the only component that knows the S3
credentials) and writes/reads an Icechunk repo through it. It uses the repo's
development object store by default:

    docker compose up -d rustfs rustfs_init
    python examples/remote_signing.py

Override with S3_ENDPOINT, S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY.

With Lakekeeper the signer URL and token come from the catalog instead: load a table
whose location covers the Icechunk repo prefix, read ``s3.signer.uri`` and
``s3.signer.endpoint`` from the ``config`` of the load-table response, and use your
catalog OAuth token. Pass ``get_remote_signer_token`` instead of
``remote_signer_token`` when the token expires (e.g. an OAuth client-credentials
flow), as shown in the second part of this script.
"""

import json
import os
import threading
import urllib.request
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from botocore.auth import S3SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

import icechunk as ic
import zarr

ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:4200")
BUCKET = os.environ.get("S3_BUCKET", "testbucket")
REGION = os.environ.get("S3_REGION", "us-east-1")
# Only the signer knows these
S3_CREDS = Credentials(
    os.environ.get("S3_ACCESS_KEY", "test123"),
    os.environ.get("S3_SECRET_KEY", "test123"),
)
TOKEN = "catalog-token"
# Tokens issued by the mock token endpoint (/token) that the signer accepts
ISSUED_TOKENS: set[str] = set()
TOKEN_URL = ""


class UnsignedPayloadSigV4(S3SigV4Auth):
    """The signer never sees the body, so use the client's UNSIGNED-PAYLOAD."""

    def _should_sha256_sign_payload(self, request: AWSRequest) -> bool:
        return False


class MockSigner(BaseHTTPRequestHandler):
    calls: list[str] = []
    token_fetches = 0

    def do_POST(self) -> None:
        if self.path == "/token":
            # Stand-in for the catalog's OAuth token endpoint
            MockSigner.token_fetches += 1
            token = os.urandom(8).hex()
            ISSUED_TOKENS.add(token)
            self.send_json({"access_token": token, "expires_in": 3600})
            return
        auth = self.headers.get("Authorization", "")
        if auth not in (f"Bearer {TOKEN}", *(f"Bearer {t}" for t in ISSUED_TOKENS)):
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.calls.append(f"{req['method']} {req['uri']}")
        # A real catalog would authorize the request here, e.g. check that the URI
        # is inside a location this token may access, and read vs write permission.
        headers = {k: ",".join(v) for k, v in req["headers"].items()}
        aws_req = AWSRequest(method=req["method"], url=req["uri"], headers=headers)
        UnsignedPayloadSigV4(S3_CREDS, "s3", req["region"]).add_auth(aws_req)
        self.send_json(
            {"uri": req["uri"], "headers": {k: [v] for k, v in aws_req.headers.items()}}
        )

    def send_json(self, obj: object) -> None:
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


def fetch_token() -> ic.S3SignerToken:
    """Get a fresh catalog token. Must be pickleable, so a module-level function."""
    req = urllib.request.Request(TOKEN_URL, method="POST")
    with urllib.request.urlopen(req) as resp:
        body = json.load(resp)
    expires = datetime.now(UTC) + timedelta(seconds=body["expires_in"])
    return ic.S3SignerToken(body["access_token"], expires_after=expires)


def write_and_read(storage: ic.Storage) -> None:
    repo = ic.Repository.create(storage)
    session = repo.writable_session("main")
    root = zarr.group(session.store)
    root.create_array("x", shape=(10,), chunks=(5,), dtype="i4")[:] = np.arange(10)
    session.commit("written via remote signer")

    repo = ic.Repository.open(storage)
    arr = zarr.open_group(repo.readonly_session("main").store, mode="r")["x"]
    assert isinstance(arr, zarr.Array)
    data = np.asarray(arr[:])
    assert (data == np.arange(10)).all(), data


def main() -> None:
    global TOKEN_URL
    signer = ThreadingHTTPServer(("127.0.0.1", 0), MockSigner)
    threading.Thread(target=signer.serve_forever, daemon=True).start()
    signer_url = f"http://127.0.0.1:{signer.server_port}/v1/aws/s3/sign"
    TOKEN_URL = f"http://127.0.0.1:{signer.server_port}/token"
    prefix = f"remote-signing-example/{os.urandom(4).hex()}"

    def storage(prefix: str, **kwargs: object) -> ic.Storage:
        return ic.s3_storage(
            bucket=BUCKET,
            prefix=prefix,
            region=REGION,
            endpoint_url=ENDPOINT,
            allow_http=True,
            force_path_style=True,
            **kwargs,  # type: ignore[arg-type]
        )

    # 1. Fixed token
    write_and_read(
        storage(prefix, remote_signer_url=signer_url, remote_signer_token=TOKEN)
    )
    print(f"fixed token: OK, signer signed {len(MockSigner.calls)} requests:")
    for call in MockSigner.calls:
        print("  ", call)

    # 2. Refreshing token: fetched on first use, then cached until it expires or
    #    the signer rejects it
    MockSigner.calls.clear()
    refreshing = storage(
        prefix + "-refresh",
        remote_signer_url=signer_url,
        get_remote_signer_token=fetch_token,
    )
    write_and_read(refreshing)
    print(
        f"refreshing token: OK, {len(MockSigner.calls)} signed requests, "
        f"{MockSigner.token_fetches} token fetch(es)"
    )
    ISSUED_TOKENS.clear()  # the catalog revokes all tokens
    repo = ic.Repository.open(refreshing)
    repo.readonly_session("main")
    print(
        f"after revocation: OK, {MockSigner.token_fetches} token fetches in total "
        "(a 401 triggers one refetch)"
    )

    # Without the signer, the same repo is not accessible.
    try:
        ic.Repository.open(storage(prefix, anonymous=True))
        print("note: anonymous access also worked, the bucket is public")
    except ic.IcechunkError:
        print("anonymous access denied, as expected")
    signer.shutdown()


if __name__ == "__main__":
    main()
