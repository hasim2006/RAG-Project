# [Doc 202] IT Equipment, Laptop Hardware & Cyber Security Policy
**Document Version:** 4.2 | **Owner:** Information Security Team

## Section 1: Standard Hardware Provisioning
Upon joining the organization, every software engineer, product manager, and designer is issued hardware based on their role tier:

- **Engineering & Architecture:** Apple MacBook Pro 16" (M3 Max, 36GB RAM) or Dell Precision Workstation (64GB RAM).
- **General Operations:** Apple MacBook Air 15" (M3, 16GB RAM) or Lenovo ThinkPad X1 Carbon.
- **Refresh Cycle:** Laptops are refreshed every 36 months (3 years). Upon completion of 3 years, employees may choose to keep their legacy machine for a nominal $50 fee or turn it in for safe recycling.

## Section 2: Security & Encryption Protocols
1. **Full-Disk Encryption:** FileVault (macOS) or BitLocker (Windows) must remain enabled at all times with strict recovery keys managed by Okta MDM.
2. **Multi-Factor Authentication (MFA):** Hardware security keys (YubiKey) are required for all access to AWS, GitHub, and production Kubernetes clusters.
3. **Password Vaulting:** All corporate logins must use 1Password. Plaintext passwords stored in local files, Slack messages, or Git repos are flagged automatically and trigger a security escalation.
4. **VPN Requirements:** All public network connections (airports, cafes, hotels) must connect via Cloudflare Zero Trust VPN prior to accessing internal administrative APIs.
