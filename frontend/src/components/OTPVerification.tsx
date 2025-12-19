import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Alert,
  Paper,
  CircularProgress,
  Link,
  Grid,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import apiService from '../services/api';
import { useAuth } from '../contexts/AuthContext';

// Styled OTP input field
const OTPInput = styled(TextField)(({ theme }) => ({
  '& .MuiOutlinedInput-root': {
    width: '60px',
    height: '60px',
    fontSize: '24px',
    fontWeight: 'bold',
    textAlign: 'center',
    '& input': {
      textAlign: 'center',
      padding: '16px 0',
    },
  },
}));

interface OTPVerificationProps {
  email: string;
  onVerificationSuccess?: (token: string, user: any) => void;
  onBack?: () => void;
}

const OTPVerification: React.FC<OTPVerificationProps> = ({
  email,
  onVerificationSuccess,
  onBack,
}) => {
  const { loginWithToken } = useAuth();
  const [otp, setOtp] = useState(['', '', '', '', '', '']);
  const [loading, setLoading] = useState(false);
  const [resending, setResending] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [timeLeft, setTimeLeft] = useState(600); // 10 minutes in seconds
  
  const inputRefs = useRef<(HTMLInputElement | null)[]>([]);

  // Countdown timer
  useEffect(() => {
    if (timeLeft > 0) {
      const timer = setTimeout(() => setTimeLeft(timeLeft - 1), 1000);
      return () => clearTimeout(timer);
    }
  }, [timeLeft]);

  // Format time display
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Handle OTP input change
  const handleOTPChange = (index: number, value: string) => {
    if (value.length > 1) return; // Only allow single digit
    if (!/^\d*$/.test(value)) return; // Only allow numbers

    const newOtp = [...otp];
    newOtp[index] = value;
    setOtp(newOtp);

    // Auto-focus next input
    if (value && index < 5) {
      inputRefs.current[index + 1]?.focus();
    }

    // Auto-submit when all fields are filled
    if (newOtp.every(digit => digit !== '') && newOtp.join('').length === 6) {
      handleVerifyOTP(newOtp.join(''));
    }
  };

  // Handle backspace
  const handleKeyDown = (index: number, e: React.KeyboardEvent) => {
    if (e.key === 'Backspace' && !otp[index] && index > 0) {
      inputRefs.current[index - 1]?.focus();
    }
  };

  // Handle paste
  const handlePaste = (e: React.ClipboardEvent) => {
    e.preventDefault();
    const pastedData = e.clipboardData.getData('text').replace(/\D/g, '').slice(0, 6);
    
    if (pastedData.length === 6) {
      const newOtp = pastedData.split('');
      setOtp(newOtp);
      handleVerifyOTP(pastedData);
    }
  };

  // Verify OTP
  const handleVerifyOTP = async (otpCode?: string) => {
    const codeToVerify = otpCode || otp.join('');
    
    if (codeToVerify.length !== 6) {
      setError('Please enter all 6 digits');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await apiService.verifyOTP(email, codeToVerify);
      
      setSuccess('Email verified successfully!');
      
      // Use AuthContext to login or call success callback
      if (onVerificationSuccess) {
        onVerificationSuccess(response.access_token, response.user);
      } else {
        // Direct login and redirect to dashboard
        loginWithToken(response.access_token, response.user);
      }
      
    } catch (err: any) {
      setError(err.message);
      
      // Clear OTP on error
      setOtp(['', '', '', '', '', '']);
      inputRefs.current[0]?.focus();
    } finally {
      setLoading(false);
    }
  };

  // Resend OTP
  const handleResendOTP = async () => {
    setResending(true);
    setError('');
    setSuccess('');

    try {
      const response = await apiService.resendOTP(email);
      setSuccess(response.message);
      setTimeLeft(600); // Reset timer
      setOtp(['', '', '', '', '', '']); // Clear current OTP
      inputRefs.current[0]?.focus();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setResending(false);
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 4, maxWidth: 500, mx: 'auto' }}>
      <Box textAlign="center" mb={4}>
        <Typography variant="h4" gutterBottom fontWeight={600} color="primary">
          📧 Verify Your Email
        </Typography>
        <Typography variant="body1" color="textSecondary" mb={2}>
          We've sent a 6-digit code to:
        </Typography>
        <Typography variant="h6" fontWeight={600} color="primary">
          {email}
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 3 }}>
          {success}
        </Alert>
      )}

      <Box mb={4}>
        <Typography variant="body2" textAlign="center" mb={2} color="textSecondary">
          Enter the 6-digit code:
        </Typography>
        
        <Grid container spacing={1} justifyContent="center" onPaste={handlePaste}>
          {otp.map((digit, index) => (
            <Grid item key={index}>
              <OTPInput
                inputRef={(el) => (inputRefs.current[index] = el)}
                value={digit}
                onChange={(e) => handleOTPChange(index, e.target.value)}
                onKeyDown={(e) => handleKeyDown(index, e)}
                variant="outlined"
                inputProps={{
                  maxLength: 1,
                  style: { textAlign: 'center' }
                }}
                disabled={loading}
              />
            </Grid>
          ))}
        </Grid>
      </Box>

      <Box textAlign="center" mb={3}>
        <Button
          variant="contained"
          size="large"
          onClick={() => handleVerifyOTP()}
          disabled={loading || otp.some(digit => digit === '')}
          sx={{ minWidth: 200 }}
        >
          {loading ? (
            <>
              <CircularProgress size={20} sx={{ mr: 1 }} />
              Verifying...
            </>
          ) : (
            'Verify Email'
          )}
        </Button>
      </Box>

      <Box textAlign="center" mb={2}>
        {timeLeft > 0 ? (
          <Typography variant="body2" color="textSecondary">
            Code expires in: <strong>{formatTime(timeLeft)}</strong>
          </Typography>
        ) : (
          <Typography variant="body2" color="error">
            Code has expired
          </Typography>
        )}
      </Box>

      <Box textAlign="center">
        <Typography variant="body2" color="textSecondary" mb={1}>
          Didn't receive the code?
        </Typography>
        <Link
          component="button"
          variant="body2"
          onClick={handleResendOTP}
          disabled={resending || timeLeft > 540} // Allow resend after 1 minute
          sx={{ cursor: resending ? 'not-allowed' : 'pointer' }}
        >
          {resending ? 'Sending...' : 'Resend Code'}
        </Link>
        
        {onBack && (
          <>
            <Typography variant="body2" color="textSecondary" component="span" mx={1}>
              •
            </Typography>
            <Link
              component="button"
              variant="body2"
              onClick={onBack}
              sx={{ cursor: 'pointer' }}
            >
              Back to Signup
            </Link>
          </>
        )}
      </Box>

      <Box mt={3} p={2} bgcolor="grey.50" borderRadius={1}>
        <Typography variant="caption" color="textSecondary" textAlign="center" display="block">
          💡 Tip: You can paste the entire 6-digit code at once
        </Typography>
      </Box>
    </Paper>
  );
};

export default OTPVerification;