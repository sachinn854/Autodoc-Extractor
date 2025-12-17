import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import {
  Container,
  Box,
  Typography,
  Paper,
  Grid,
  Alert,
  Snackbar,
  Card,
  CardContent,
  Chip,
  AppBar,
  Toolbar,
  Button,
  IconButton,
  Menu,
  MenuItem,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Analytics as AnalyticsIcon,
  TableChart as TableIcon,
  Speed as SpeedIcon,
  Security as SecurityIcon,
  AutoFixHigh as AIIcon,
  AccountCircle,
  Dashboard as DashboardIcon,
} from '@mui/icons-material';
import FileUpload from '../components/FileUpload';
import Loader from '../components/Loader';
import { UploadResponse } from '../types/schema';
import { useAuth } from '../contexts/AuthContext';

const HomePage: React.FC = () => {
  const router = useRouter();
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string>('');
  const [showSuccess, setShowSuccess] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const { user, token, logout, isLoading } = useAuth();

  useEffect(() => {
    if (!isLoading && !token) {
      router.push('/login');
    }
  }, [token, isLoading, router]);

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    handleMenuClose();
    logout();
  };

  const handleUploadComplete = (response: UploadResponse) => {
    setIsProcessing(true);
    
    // Redirect to results page
    setTimeout(() => {
      router.push(`/result/${response.job_id}`);
    }, 1000);
  };

  const handleUploadError = (errorMessage: string) => {
    setError(errorMessage);
    setIsProcessing(false);
  };

  const features = [
    {
      icon: <AIIcon sx={{ fontSize: 40 }} color="primary" />,
      title: 'AI-Powered OCR',
      description: 'Advanced machine learning algorithms for accurate text extraction from receipts and invoices.'
    },
    {
      icon: <TableIcon sx={{ fontSize: 40 }} color="primary" />,
      title: 'Smart Table Detection',
      description: 'Automatically detects and extracts structured data from complex table layouts.'
    },
    {
      icon: <AnalyticsIcon sx={{ fontSize: 40 }} color="primary" />,
      title: 'Intelligent Insights',
      description: 'Generate spending analytics, category breakdowns, and anomaly detection automatically.'
    },
    {
      icon: <SpeedIcon sx={{ fontSize: 40 }} color="primary" />,
      title: 'Fast Processing',
      description: 'Process documents in seconds with our optimized pipeline and real-time progress updates.'
    },
    {
      icon: <SecurityIcon sx={{ fontSize: 40 }} color="primary" />,
      title: 'Secure & Private',
      description: 'Your documents are processed securely and automatically cleaned up after processing.'
    }
  ];

  if (isProcessing) {
    return (
      <Container maxWidth="md" sx={{ py: 8 }}>
        <Head>
          <title>Processing Document - AutoDoc Extractor</title>
        </Head>
        
        <Paper elevation={3} sx={{ p: 6, textAlign: 'center' }}>
          <Typography variant="h4" gutterBottom color="primary" fontWeight={600}>
            🚀 Processing Your Document
          </Typography>
          <Typography variant="body1" color="textSecondary" mb={4}>
            Please wait while we extract and analyze your document data...
          </Typography>
          
          <Loader 
            message="Initializing document processing pipeline..." 
            size="large"
          />
          
          <Alert severity="info" sx={{ mt: 4, textAlign: 'left' }}>
            <Typography variant="body2">
              <strong>What's happening:</strong><br/>
              • Document upload completed<br/>
              • OCR and table detection in progress<br/>
              • ML categorization and insights generation<br/>
              • Results will be available shortly
            </Typography>
          </Alert>
        </Paper>
      </Container>
    );
  }

  return (
    <>
      <Head>
        <title>AutoDoc Extractor - AI Document Processing & Insights</title>
        <meta name="description" content="Upload receipts and invoices to automatically extract data and get intelligent insights with AI-powered OCR and table detection." />
      </Head>

      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Autodoc Extractor
          </Typography>
          <Button
            color="inherit"
            startIcon={<DashboardIcon />}
            onClick={() => router.push('/dashboard')}
            sx={{ mr: 2 }}
          >
            Dashboard
          </Button>
          <Typography variant="body1" sx={{ mr: 2 }}>
            {user?.email}
          </Typography>
          <IconButton color="inherit" onClick={handleMenuOpen}>
            <AccountCircle />
          </IconButton>
          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleMenuClose}
          >
            <MenuItem onClick={handleLogout}>Logout</MenuItem>
          </Menu>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Hero Section */}
        <Box textAlign="center" mb={8}>
          <Typography 
            variant="h2" 
            component="h1" 
            gutterBottom 
            fontWeight={700}
            sx={{ 
              background: 'linear-gradient(45deg, #3b82f6, #8b5cf6)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            📄 AutoDoc Extractor
          </Typography>
          
          <Typography variant="h5" color="textSecondary" mb={4} fontWeight={300}>
            Transform your documents into actionable insights with AI
          </Typography>
          
          <Box display="flex" justifyContent="center" gap={2} mb={6} flexWrap="wrap">
            <Chip label="OCR Technology" color="primary" />
            <Chip label="Table Detection" color="primary" />
            <Chip label="ML Insights" color="primary" />
            <Chip label="Real-time Processing" color="primary" />
          </Box>
        </Box>

        {/* Upload Section */}
        <Box mb={8}>
          <Typography variant="h4" textAlign="center" mb={4} fontWeight={600}>
            Upload Your Document
          </Typography>
          
          <FileUpload
            onUploadComplete={handleUploadComplete}
            onUploadError={handleUploadError}
            isLoading={isProcessing}
            acceptedTypes={['image/jpeg', 'image/png', 'image/tiff', 'application/pdf']}
            maxSize={10}
          />
        </Box>

        {/* Features Section */}
        <Box mb={8}>
          <Typography variant="h4" textAlign="center" mb={6} fontWeight={600}>
            ⚡ Powerful Features
          </Typography>
          
          <Grid container spacing={4}>
            {features.map((feature, index) => (
              <Grid item xs={12} md={6} lg={4} key={index}>
                <Card 
                  elevation={2}
                  sx={{ 
                    height: '100%',
                    transition: 'transform 0.2s ease-in-out',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: 4
                    }
                  }}
                >
                  <CardContent sx={{ textAlign: 'center', p: 4 }}>
                    <Box mb={2}>
                      {feature.icon}
                    </Box>
                    <Typography variant="h6" gutterBottom fontWeight={600}>
                      {feature.title}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      {feature.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* Process Flow */}
        <Paper elevation={1} sx={{ p: 4, backgroundColor: '#f8fafc' }}>
          <Typography variant="h5" textAlign="center" mb={4} fontWeight={600}>
            📋 How It Works
          </Typography>
          
          <Grid container spacing={3} textAlign="center">
            <Grid item xs={12} sm={6} md={3}>
              <Box>
                <Typography variant="h4" color="primary" fontWeight={700}>
                  1️⃣
                </Typography>
                <Typography variant="h6" gutterBottom>
                  Upload
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Drop your receipt, invoice, or document
                </Typography>
              </Box>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Box>
                <Typography variant="h4" color="primary" fontWeight={700}>
                  2️⃣
                </Typography>
                <Typography variant="h6" gutterBottom>
                  Extract
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  AI detects tables and extracts data
                </Typography>
              </Box>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Box>
                <Typography variant="h4" color="primary" fontWeight={700}>
                  3️⃣
                </Typography>
                <Typography variant="h6" gutterBottom>
                  Analyze
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Generate insights and categorization
                </Typography>
              </Box>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Box>
                <Typography variant="h4" color="primary" fontWeight={700}>
                  4️⃣
                </Typography>
                <Typography variant="h6" gutterBottom>
                  Export
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Download CSV or share results
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* Supported Formats */}
        <Box mt={6} textAlign="center">
          <Typography variant="h6" mb={2} fontWeight={600}>
            📎 Supported Formats
          </Typography>
          <Box display="flex" justifyContent="center" gap={2} flexWrap="wrap">
            <Chip label="JPEG Images" variant="outlined" />
            <Chip label="PNG Images" variant="outlined" />
            <Chip label="TIFF Images" variant="outlined" />
            <Chip label="PDF Documents" variant="outlined" />
          </Box>
          <Typography variant="body2" color="textSecondary" mt={2}>
            Maximum file size: 10MB
          </Typography>
        </Box>
      </Container>

      {/* Error Snackbar */}
      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={() => setError('')}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={() => setError('')} 
          severity="error" 
          sx={{ width: '100%' }}
        >
          {error}
        </Alert>
      </Snackbar>

      {/* Success Snackbar */}
      <Snackbar
        open={showSuccess}
        autoHideDuration={3000}
        onClose={() => setShowSuccess(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={() => setShowSuccess(false)} 
          severity="success" 
          sx={{ width: '100%' }}
        >
          Upload successful! Redirecting to results...
        </Alert>
      </Snackbar>
    </>
  );
};

export default HomePage;