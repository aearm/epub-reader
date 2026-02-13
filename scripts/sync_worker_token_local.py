#!/usr/bin/env python3
import json
import os
import sys
import time
import urllib.error
import urllib.request


def env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


COGNITO_REGION = env("COGNITO_REGION", "eu-west-1")
COGNITO_CLIENT_ID = env("COGNITO_CLIENT_ID")
COGNITO_USERNAME = env("COGNITO_USERNAME")
COGNITO_PASSWORD = env("COGNITO_PASSWORD")
WORKER_TOKEN_URL = env("WORKER_TOKEN_URL", "http://127.0.0.1:5001/worker/token")
REFRESH_INTERVAL_SECONDS = max(60, int(env("TOKEN_REFRESH_INTERVAL_SECONDS", str(45 * 60))))


def post_json(url: str, payload: dict, headers: dict[str, str], timeout: int = 20) -> dict:
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def cognito_auth(auth_flow: str, auth_parameters: dict[str, str]) -> dict:
    url = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/"
    headers = {
        "Content-Type": "application/x-amz-json-1.1",
        "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
    }
    payload = {
        "AuthFlow": auth_flow,
        "ClientId": COGNITO_CLIENT_ID,
        "AuthParameters": auth_parameters,
    }
    return post_json(url, payload, headers=headers, timeout=20)


def sync_worker_token(access_token: str) -> None:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    payload = {"access_token": access_token}
    result = post_json(WORKER_TOKEN_URL, payload, headers=headers, timeout=20)
    print(f"worker token sync ok: {result.get('success')}", flush=True)


def main() -> int:
    if not COGNITO_CLIENT_ID:
        print("Missing env: COGNITO_CLIENT_ID", file=sys.stderr)
        return 2
    if not COGNITO_USERNAME:
        print("Missing env: COGNITO_USERNAME", file=sys.stderr)
        return 2
    if not COGNITO_PASSWORD:
        print("Missing env: COGNITO_PASSWORD", file=sys.stderr)
        return 2

    refresh_token = None
    access_token = None

    while True:
        try:
            if not access_token:
                response = cognito_auth(
                    "USER_PASSWORD_AUTH",
                    {"USERNAME": COGNITO_USERNAME, "PASSWORD": COGNITO_PASSWORD},
                )
                auth = response.get("AuthenticationResult") or {}
                access_token = auth.get("AccessToken")
                refresh_token = auth.get("RefreshToken")
                if not access_token:
                    raise RuntimeError(f"No AccessToken in response: {response}")
            else:
                if refresh_token:
                    response = cognito_auth(
                        "REFRESH_TOKEN_AUTH",
                        {"REFRESH_TOKEN": refresh_token},
                    )
                    auth = response.get("AuthenticationResult") or {}
                    access_token = auth.get("AccessToken")
                    if not access_token:
                        raise RuntimeError(f"No AccessToken in refresh response: {response}")
                else:
                    access_token = None
                    continue

            sync_worker_token(access_token)
            time.sleep(REFRESH_INTERVAL_SECONDS)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"HTTP error {exc.code}: {body[:500]}", file=sys.stderr, flush=True)
            access_token = None
            time.sleep(10)
        except Exception as exc:
            print(f"token sync loop error: {exc}", file=sys.stderr, flush=True)
            access_token = None
            time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())
