import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import {
  Container,
  Box,
  Typography,
  Alert,
  Button,
  Paper,
  Grid,
  Chip,
  CircularProgress,
  Tabs,
  Tab,
  Snackbar,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  ArrowBack as BackIcon,
  Download as DownloadIcon,
  Share as ShareIcon,
  Refresh as RefreshIcon,
  Error as ErrorIcon,
  CheckCircle as SuccessIcon
} from '@mui/icons-material';
import EditableTable from '../../components/EditableTable';
import Loader from '../../components/Loader';
// import InsightsPanel from '../../components/InsightsPanel';
// import ImagePreview from '../../components/ImagePreview';
import { 
  ResultResponse, 
  JobStatus, 
  ExtractedResult, 
  Item 
} from '../../types/schema';
import apiService from '../../services/api';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel({ children, value, index, ...other }: TabPanelProps) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`result-tabpanel-${index}`}
      aria-labelledby={`result-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ py: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const ResultPage: React.FC = () => {
  const router = useRouter();
  const { job_id } = router.query;
  
  const [results, setResults] = useState<ResultResponse | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [activeTab, setActiveTab] = useState(0);
  const [extractedData, setExtractedData] = useState<Item[]>([]);
  const [dataModified, setDataModified] = useState(false);
  const [saving, setSaving] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  
  // Poll for job completion
  useEffect(() => {
    if (!job_id || typeof job_id !== 'string') return;

    const pollStatus = async () => {
      try {
        const status = await apiService.getJobStatus(job_id);
        setJobStatus(status);
        
        if (status.status === 'completed') {
          const resultData = await apiService.getResults(job_id);
          // Normalize API: prefer extracted_data -> extracted for UI
          const normalized = {
            ...resultData,
            extracted: (resultData as any).extracted_data || resultData.extracted
          } as ResultResponse;
          setResults(normalized);
          setExtractedData(normalized.extracted?.items || []);
          setLoading(false);
        } else if (status.status === 'failed') {
          setError(status.error || 'Processing failed');
          setLoading(false);
        } else {
          // Increase polling interval during model download
          const delay = status.progress?.includes('Downloading OCR models') ? 10000 : 3000;
          setTimeout(pollStatus, delay);
        }
      } catch (err: any) {
        // If it's a timeout during model download, retry after longer delay
        if (err.message.includes('timeout') || err.message.includes('Failed to get status')) {
          console.warn('Status check failed, retrying...', err.message);
          setTimeout(pollStatus, 10000); // Retry after 10 seconds
        } else {
          setError(err.message);
          setLoading(false);
        }
      }
    };

    pollStatus();
  }, [job_id]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleDataChange = (updatedData: Item[]) => {
    setExtractedData(updatedData);
    setDataModified(true);
  };

  const handleSaveChanges = async () => {
    if (!job_id || typeof job_id !== 'string' || !results?.extracted) return;
    
    setSaving(true);
    try {
      const updatedExtracted: ExtractedResult = {
        ...results.extracted!,
        items: extractedData,
        total: extractedData.reduce((sum, item) => sum + item.line_total, 0)
      };
      
      await apiService.updateExtractedData(job_id, updatedExtracted);
      const refreshedResults = await apiService.getResults(job_id);
      const normalized = {
        ...refreshedResults,
        extracted: (refreshedResults as any).extracted_data || refreshedResults.extracted
      } as ResultResponse;
      setResults(normalized);
      setExtractedData(normalized.extracted?.items || []);
      
      setDataModified(false);
      setShowSuccess(true);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const handleDownloadCSV = async () => {
    if (!job_id || typeof job_id !== 'string') return;
    
    try {
      const blob = await apiService.downloadCSV(job_id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `extracted_data_${job_id}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleShare = () => {
    const url = window.location.href;
    if (navigator.share) {
      navigator.share({
        title: 'AutoDoc Extractor Results',
        text: 'Check out my document processing results',
        url: url
      });
    } else {
      navigator.clipboard.writeText(url);
      setShowSuccess(true);
    }
  };

  const getStatusChip = () => {
    if (!jobStatus) return null;
    
    const statusConfig = {
      processing: { color: 'warning' as const, icon: CircularProgress, label: 'Processing' },
      completed: { color: 'success' as const, icon: SuccessIcon, label: 'Completed' },
      failed: { color: 'error' as const, icon: ErrorIcon, label: 'Failed' },
      uploaded: { color: 'info' as const, icon: CircularProgress, label: 'Uploaded' }
    };
    
    const config = statusConfig[jobStatus.status] || statusConfig.processing;
    const Icon = config.icon;
    
    return (
      <Chip
        icon={<Icon sx={{ fontSize: '16px !important' }} />}
        label={config.label}
        color={config.color}
        size="small"
      />
    );
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Paper elevation={2} sx={{ p: 4 }}>
          <Box textAlign="center">
            <Typography variant="h5" mb={3}>
              Processing Document
            </Typography>
            {jobStatus && (
              <Typography variant="body1" color="textSecondary" mb={3}>
                {jobStatus.progress}
              </Typography>
            )}
            <Loader 
              message={jobStatus?.progress || 'Processing your document...'}
              size="large"
            />
            {jobStatus?.status === 'processing' && (
              <Alert severity="info" sx={{ mt: 3, maxWidth: 400, mx: 'auto' }}>
                {jobStatus.progress?.includes('Downloading OCR models') ? (
                  <>
                    First time setup: Downloading AI models (2-3 minutes). 
                    <br />Subsequent uploads will be much faster.
                  </>
                ) : (
                  'This usually takes 30-60 seconds depending on document complexity.'
                )}
              </Alert>
            )}
          </Box>
        </Paper>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Alert severity="error" sx={{ mb: 3 }}>
          <Typography variant="h6">Processing Failed</Typography>
          <Typography>{error}</Typography>
        </Alert>
        <Button 
          startIcon={<BackIcon />} 
          onClick={() => router.push('/')}
          variant="contained"
        >
          Try Again
        </Button>
      </Container>
    );
  }

  if (!results) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Alert severity="warning">
          <Typography variant="h6">Results Not Found</Typography>
          <Typography>The requested job results could not be found.</Typography>
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
        <Box display="flex" alignItems="center" gap={2}>
          <IconButton onClick={() => router.push('/')}>
            <BackIcon />
          </IconButton>
          <Box>
            <Typography variant="h4" fontWeight={700}>
              Processing Results
            </Typography>
            <Box display="flex" alignItems="center" gap={1} mt={1}>
              <Typography variant="body2" color="textSecondary">
                Job ID: {job_id}
              </Typography>
              {getStatusChip()}
              {dataModified && (
                <Chip label="Modified" color="warning" size="small" />
              )}
            </Box>
          </Box>
        </Box>
        
        <Box display="flex" gap={1}>
          <Tooltip title="Refresh Results">
            <IconButton 
              onClick={() => window.location.reload()}
              color="primary"
            >
              <RefreshIcon />
            </IconButton>
          </Tooltip>
          
          <Tooltip title="Share Results">
            <IconButton onClick={handleShare} color="primary">
              <ShareIcon />
            </IconButton>
          </Tooltip>
          
          {results.csv_url && (
            <Button
              startIcon={<DownloadIcon />}
              onClick={handleDownloadCSV}
              variant="outlined"
              size="small"
            >
              Download CSV
            </Button>
          )}
          
          {dataModified && (
            <Button
              onClick={handleSaveChanges}
              variant="contained"
              size="small"
              disabled={saving}
            >
              {saving ? 'Saving...' : 'Save Changes'}
            </Button>
          )}
        </Box>
      </Box>

      {/* Summary Stats */}
      {results?.extracted && (
        <Paper elevation={1} sx={{ p: 3, mb: 4, backgroundColor: '#f8fafc' }}>
          <Grid container spacing={3} textAlign="center">
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="h4" color="primary" fontWeight={700}>
                {results.extracted.items.length}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Items Extracted
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="h4" color="success.main" fontWeight={700}>
                {
                  // ALWAYS compute from current extractedData state (live calculation)
                  `$${extractedData.reduce((sum, item) => sum + (Number(item.line_total) || 0), 0).toFixed(2)}`
                }
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Total Amount
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="h5" fontWeight={600}>
                {results.extracted.vendor || 'Unknown'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Vendor
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="h5" fontWeight={600}>
                {results.extracted.date || 'Unknown'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Date
              </Typography>
            </Grid>
          </Grid>
        </Paper>
      )}

      {/* Tabs */}
      <Paper elevation={2}>
        <Tabs 
          value={activeTab} 
          onChange={handleTabChange}
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label="📊 Data Table" />
          <Tab label="📈 Insights" disabled={!results.insights} />
          <Tab label="🖼️ Preview" disabled={!results.files?.['processed_image.jpg']} />
        </Tabs>
         
        <TabPanel value={activeTab} index={0}>
          {results?.extracted ? (
            <EditableTable
              data={extractedData}
              onDataChange={handleDataChange}
              onSave={handleSaveChanges}
              isLoading={saving}
            />
          ) : (
            <Alert severity="info">
              No data was extracted from the document. This could mean:
              <ul>
                <li>No tables were detected in the document</li>
                <li>The image quality was too low for accurate OCR</li>
                <li>The document format is not supported</li>
              </ul>
            </Alert>
          )}
        </TabPanel>
        
        <TabPanel value={activeTab} index={1}>
          <Alert severity="info">
            <Typography variant="h6" mb={2}>📈 Insights Panel</Typography>
            <Typography>Insights feature temporarily disabled for debugging.</Typography>
            {results.insights && (
              <Typography variant="body2" mt={2}>
                Raw insights data available: {JSON.stringify(results.insights).substring(0, 100)}...
              </Typography>
            )}
          </Alert>
        </TabPanel>
        
        <TabPanel value={activeTab} index={2}>
          <Box>
            <Typography variant="h6" mb={2}>🖼️ Image Preview</Typography>
            {results.files?.['processed_image.jpg'] ? (
              <Paper elevation={2} sx={{ p: 2 }}>
                <img
                  src={results.files['processed_image.jpg']}
                  alt="Document Preview"
                  style={{ maxWidth: '100%', maxHeight: '500px', objectFit: 'contain' }}
                />
                <Typography variant="body2" mt={1} color="textSecondary">
                  Document processed successfully
                </Typography>
              </Paper>
            ) : (
              <Alert severity="info">
                Image preview is not available. The processed image may not have been saved.
              </Alert>
            )}
          </Box>
        </TabPanel>
      </Paper>

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
          Changes saved successfully!
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default ResultPage;
