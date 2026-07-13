# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Taiji, please report it responsibly.

**Do not open a public GitHub issue.** Instead, email the maintainers directly.

We will acknowledge your report within 48 hours and provide a timeline for remediation within 1 week.

## Supported Versions

| Version | Supported          |
|---------|-------------------|
| 1.6.x   | ✅ Active          |
| < 1.6   | ❌ No longer supported |

## Security Best Practices for Deployments

1. **API Authentication**: Set `TAIJI_API_KEYS` in your environment to enable API key validation. Never commit `.env` files with real keys.

2. **JWT Secret**: The JWT secret is auto-generated at first startup and stored in `security/.jwt_secret`. This file is excluded from version control. Rotate it periodically in production.

3. **Docker**: If using `docker-compose.yml`, change the Grafana admin password via the `GRAFANA_ADMIN_PASSWORD` environment variable. The default is for development only.

4. **Model Files**: Checkpoint files (`.pt`, `.safetensors`) contain the full model state. Treat them as sensitive artifacts.

5. **Dependencies**: Run `pip-audit` or `safety check` regularly to scan for known vulnerabilities in Python dependencies.

## Responsible Disclosure

We follow a coordinated disclosure process:
- Reporter submits vulnerability privately
- Maintainers verify and develop a fix
- Fix is released as a patch version
- Public disclosure after the fix is available
