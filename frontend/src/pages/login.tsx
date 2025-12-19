import React, { useState } from 'react';
import { useRouter } from 'next/router';
import {
  Container,
  Box,
  Paper,
  TextField,
  Button,
  Typography,
  Alert,
  Link,
  CircularProgress,
} from '@mui/material';
import { useAuth } from '../contexts/AuthContext';
import OTPVerification from '../components/OTPVerification';
import apiService from '../services/api';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showOTPVerification, setShowOTPVerification] = useState(false);
  const { login, loginWithToken } = useAuth();
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await login(email, password);
      // Redirect handled by AuthContext
    } catch (err: any) {
      const errorMessage = err.message || 'Login failed. Please try again.';
      
      // Check if error is about email verification
      if (errorMessage.includes('Email not verified') || errorMessage.includes('not verified')) {
        setShowOTPVerification(true);
        setError('Please verify your email first. We\'ll send you a new OTP code.');
        
        // Trigger OTP resend
        try {
          await apiService.resendOTP(email);
        } catch (resendErr) {
          console.error('Failed to resend OTP:', resendErr);
        }
      } else {
        setError(errorMessage);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleOTPVerificationSuccess = (token: string, user: any) => {
    // Login user and redirect
    loginWithToken(token, user);
  };

  const handleBackToLogin = () => {
    setShowOTPVerification(false);
    setError('');
  };

  if (showOTPVerification) {
    return (
      <Container maxWidth="sm">
        <Box
          sx={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <OTPVerification
            email={email}
            onVerificationSuccess={handleOTPVerificationSuccess}
            onBack={handleBackToLogin}
          />
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="sm">
      <Box
        sx={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Paper elevation={3} sx={{ p: 4, width: '100%' }}>
          <Typography variant="h4" component="h1" gutterBottom align="center">
            Autodoc Extractor
          </Typography>
          <Typography variant="h6" gutterBottom align="center" color="text.secondary">
            Login to Your Account
          </Typography>

          {error && (
            <Alert severity="error" sx={{ mt: 2, mb: 2 }}>
              {error}
            </Alert>
          )}

          <form onSubmit={handleSubmit}>
            <TextField
              fullWidth
              label="Email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              margin="normal"
              required
              autoComplete="email"
              autoFocus
            />
            <TextField
              fullWidth
              label="Password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              margin="normal"
              required
              autoComplete="current-password"
            />
            <Button
              type="submit"
              fullWidth
              variant="contained"
              size="large"
              sx={{ mt: 3, mb: 2 }}
              disabled={loading}
            >
              {loading ? <CircularProgress size={24} /> : 'Login'}
            </Button>
          </form>

          <Box sx={{ mt: 2, textAlign: 'center' }}>
            <Typography variant="body2">
              Don't have an account?{' '}
              <Link
                component="button"
                variant="body2"
                onClick={() => router.push('/signup')}
                sx={{ cursor: 'pointer' }}
              >
                Sign Up
              </Link>
            </Typography>
          </Box>
        </Paper>
      </Box>
    </Container>
  );
}
