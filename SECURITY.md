# Security policy

## Reporting a vulnerability

Please do **not** open a public issue for security problems. Use GitHub's private reporting instead: go to the **Security** tab of this repository and choose **Report a vulnerability**.

Include what you found, how to reproduce it, and the version or commit. We will acknowledge reports as soon as we can; this is a volunteer project, so we cannot promise a fixed response time.

## Scope notes

- API keys are read from `.env` or Windows Credential Manager. Never post keys in issues or logs. If you think a key leaked, revoke it at the provider first.
- Transcribed text is sent to the configured translation providers (Groq, DeepL). Audio is not uploaded.
