import { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import {
  Container,
  Paper,
  Typography,
  CircularProgress,
  Box,
  Button,
  Alert
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import axios from 'axios';

// API URL: Use environment variable or same-origin in production
const API_URL = process.env.NEXT_PUBLIC_API_URL || 
  (process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8001');

export default function VerifyEmail() {
  const router = useRouter();
  const { token } = router.query;
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [message, setMessage] = useState('');

  useEffect(() => {
    if (token) {
      verifyEmail(token as string);
    }
  }, [token]);

  const verifyEmail = async (verificationToken: string) => {
    try {
      const response = await axios.get(`${API_URL}/auth/verify-email`, {
        params: { token: verificationToken }
      });

      setStatus('success');
      setMessage(response.data.message || 'Email verified successfully!');
    } catch (error: any) {
      setStatus('error');
      setMessage(
        error.response?.data?.detail || 
        'Verification failed. The link may be invalid or expired.'
      );
    }
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 8 }}>
      <Paper elevation={3} sx={{ p: 4, textAlign: 'center' }}>
        {status === 'loading' && (
          <Box>
            <CircularProgress size={60} sx={{ mb: 3 }} />
            <Typography variant="h5" gutterBottom>
              Verifying your email...
            </Typography>
            <Typography color="text.secondary">
              Please wait while we verify your email address.
            </Typography>
          </Box>
        )}

        {status === 'success' && (
          <Box>
            <CheckCircleIcon 
              sx={{ fontSize: 80, color: 'success.main', mb: 2 }} 
            />
            <Typography variant="h4" gutterBottom color="success.main">
              Email Verified!
            </Typography>
            <Alert severity="success" sx={{ mb: 3 }}>
              {message}
            </Alert>
            <Typography color="text.secondary" paragraph>
              Your email has been successfully verified. You can now login to your account.
            </Typography>
            <Button
              variant="contained"
              size="large"
              onClick={() => router.push('/login')}
              sx={{ mt: 2 }}
            >
              Go to Login
            </Button>
          </Box>
        )}

        {status === 'error' && (
          <Box>
            <ErrorIcon 
              sx={{ fontSize: 80, color: 'error.main', mb: 2 }} 
            />
            <Typography variant="h4" gutterBottom color="error">
              Verification Failed
            </Typography>
            <Alert severity="error" sx={{ mb: 3 }}>
              {message}
            </Alert>
            <Typography color="text.secondary" paragraph>
              The verification link may be invalid or has already been used.
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', mt: 3 }}>
              <Button
                variant="outlined"
                onClick={() => router.push('/signup')}
              >
                Sign Up Again
              </Button>
              <Button
                variant="contained"
                onClick={() => router.push('/login')}
              >
                Try Login
              </Button>
            </Box>
          </Box>
        )}
      </Paper>
    </Container>
  );
}
