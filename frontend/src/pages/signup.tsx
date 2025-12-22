import React, { useState } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import {
  Container,
  Box,
  Typography,
  TextField,
  Button,
  Alert,
  Paper,
  Link,
  CircularProgress,
  Divider,
} from '@mui/material';
import { Person, Email, Lock } from '@mui/icons-material';
import apiService from '../services/api';
import { useAuth } from '../contexts/AuthContext';

const SignupPage: React.FC = () => {
  const router = useRouter();
  const { loginWithToken } = useAuth();
  
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    fullName: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await apiService.signup(
        formData.email,
        formData.password,
        formData.fullName
      );

      // Direct login after signup (no verification needed)
      loginWithToken(response.access_token, response.user);
      router.push('/dashboard');
      
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleOTPVerificationSuccess = (token: string, user: any) => {
    // Login user and redirect to dashboard
    loginWithToken(token, user);
    router.push('/dashboard');
  };

  const handleBackToSignup = () => {
    setStep('signup');
    setError('');
  };

  return (
    <>
      <Head>
        <title>Sign Up - Autodoc Extractor</title>
        <meta name="description" content="Create your account to start processing documents with AI" />
      </Head>

      <Container maxWidth="sm" sx={{ py: 8 }}>
        <Paper elevation={3} sx={{ p: 4 }}>
          <Box textAlign="center" mb={4}>
            <Typography variant="h4" gutterBottom fontWeight={600} color="primary">
              📄 Create Account
            </Typography>
            <Typography variant="body1" color="textSecondary">
              Join Autodoc Extractor to start processing your documents
            </Typography>
          </Box>

          {error && (
            <Alert severity="error" sx={{ mb: 3 }}>
              {error}
            </Alert>
          )}

          <Box component="form" onSubmit={handleSignup}>
            <Box mb={3}>
              <TextField
                fullWidth
                label="Full Name"
                name="fullName"
                value={formData.fullName}
                onChange={handleInputChange}
                variant="outlined"
                InputProps={{
                  startAdornment: <Person sx={{ mr: 1, color: 'text.secondary' }} />,
                }}
              />
            </Box>

            <Box mb={3}>
              <TextField
                fullWidth
                label="Email Address"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleInputChange}
                variant="outlined"
                required
                InputProps={{
                  startAdornment: <Email sx={{ mr: 1, color: 'text.secondary' }} />,
                }}
              />
            </Box>

            <Box mb={4}>
              <TextField
                fullWidth
                label="Password"
                name="password"
                type="password"
                value={formData.password}
                onChange={handleInputChange}
                variant="outlined"
                required
                helperText="Minimum 6 characters"
                InputProps={{
                  startAdornment: <Lock sx={{ mr: 1, color: 'text.secondary' }} />,
                }}
              />
            </Box>

            <Button
              type="submit"
              fullWidth
              variant="contained"
              size="large"
              disabled={loading || !formData.email || !formData.password}
              sx={{ mb: 3, py: 1.5 }}
            >
              {loading ? (
                <>
                  <CircularProgress size={20} sx={{ mr: 1 }} />
                  Creating Account...
                </>
              ) : (
                'Create Account'
              )}
            </Button>

            <Divider sx={{ mb: 3 }}>
              <Typography variant="body2" color="textSecondary">
                Already have an account?
              </Typography>
            </Divider>

            <Box textAlign="center">
              <Link
                component="button"
                type="button"
                variant="body1"
                onClick={() => router.push('/login')}
                sx={{ cursor: 'pointer' }}
              >
                Sign In Instead
              </Link>
            </Box>
          </Box>

          <Box mt={4} p={2} bgcolor="grey.50" borderRadius={1}>
            <Typography variant="caption" color="textSecondary" textAlign="center" display="block">
              🔐 We'll send a 6-digit verification code to your email address
            </Typography>
          </Box>
        </Paper>
      </Container>
    </>
  );
};

export default SignupPage;