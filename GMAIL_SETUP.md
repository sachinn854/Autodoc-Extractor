# 📧 Gmail SMTP Setup for Email Verification

## 🎯 Quick Setup (5 Minutes)

### Step 1: Enable 2-Factor Authentication
1. Go to [Google Account Settings](https://myaccount.google.com/)
2. Click **Security** → **2-Step Verification**
3. Follow the setup process

### Step 2: Generate App Password
1. Go to [App Passwords](https://myaccount.google.com/apppasswords)
2. Select **Mail** and **Other (Custom name)**
3. Enter: `Autodoc Extractor`
4. Click **Generate**
5. **Copy the 16-character password** (e.g., `abcd efgh ijkl mnop`)

### Step 3: Set Environment Variables in Render

In your Render dashboard:

```bash
# Required for email verification
SMTP_EMAIL=your-gmail@gmail.com
SMTP_PASSWORD=abcd efgh ijkl mnop
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

### Step 4: Test Email Verification

1. **Restart your Render service** (to load new environment variables)
2. **Sign up with a new account**
3. **Check your email** for verification link
4. **Click the link** to verify

## 🔧 Troubleshooting

### Issue: "Authentication failed"
- **Solution**: Make sure 2FA is enabled and you're using App Password (not regular password)

### Issue: "SMTP not configured"
- **Solution**: Check environment variables are set correctly in Render dashboard

### Issue: "Email not received"
- **Solution**: Check spam folder, verify email address is correct

## 🧪 Test SMTP Configuration

Use this endpoint to test if SMTP is working:

```bash
curl -X POST "https://your-app.onrender.com/auth/test-email" \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com"}'
```

## 📱 Alternative: Use Different Email Provider

### SendGrid (Recommended for Production)
```bash
SMTP_SERVER=smtp.sendgrid.net
SMTP_PORT=587
SMTP_EMAIL=apikey
SMTP_PASSWORD=your-sendgrid-api-key
```

### Outlook/Hotmail
```bash
SMTP_SERVER=smtp-mail.outlook.com
SMTP_PORT=587
SMTP_EMAIL=your-email@outlook.com
SMTP_PASSWORD=your-app-password
```

## 🎉 Success!

Once configured, users will:
1. **Sign up** → Account created
2. **Check email** → Verification link received
3. **Click link** → Account verified
4. **Login** → Full access to upload documents

**No more manual verification needed!** ✅