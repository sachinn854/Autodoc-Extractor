import React, { useState, useCallback } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Button, 
  LinearProgress,
  Alert,
  Chip,
  IconButton
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  InsertDriveFile as FileIcon,
  Delete as DeleteIcon,
  CheckCircle as CheckIcon
} from '@mui/icons-material';
import { FileUploadProps } from '../types/schema';
import apiService from '../services/api';
import { useAuth } from '../contexts/AuthContext';

const FileUpload: React.FC<FileUploadProps> = ({
  onUploadComplete,
  onUploadError,
  isLoading = false,
  acceptedTypes = ['image/jpeg', 'image/png', 'image/tiff', 'application/pdf'],
  maxSize = 10 // 10MB default
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const { token } = useAuth();

  // Drag and drop handlers
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelection(files[0]);
    }
  }, []);

  // File selection validation
  const handleFileSelection = (file: File) => {
    // Validate file type
    if (!acceptedTypes.includes(file.type)) {
      onUploadError(`Invalid file type. Please upload: ${acceptedTypes.join(', ')}`);
      return;
    }

    // Validate file size
    const fileSizeMB = file.size / (1024 * 1024);
    if (fileSizeMB > maxSize) {
      onUploadError(`File too large. Maximum size is ${maxSize}MB`);
      return;
    }

    setSelectedFile(file);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelection(file);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    try {
      setUploadProgress(0);
      
      // Simulate upload progress (since axios doesn't provide real progress for FormData)
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      // Call API service directly
      const response = await apiService.uploadFile(selectedFile);
      
      clearInterval(progressInterval);
      setUploadProgress(100);
      
      setTimeout(() => {
        onUploadComplete(response);
        setSelectedFile(null);
        setUploadProgress(0);
      }, 500);

    } catch (error: any) {
      setUploadProgress(0);
      onUploadError(error.message || 'Upload failed');
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setUploadProgress(0);
  };

  const getFileIcon = (fileType: string) => {
    if (fileType.startsWith('image/')) {
      return '🖼️';
    }
    if (fileType === 'application/pdf') {
      return '📄';
    }
    return '📁';
  };

  return (
    <Box className="w-full max-w-2xl mx-auto">
      {/* Upload Area */}
      <Paper
        elevation={isDragOver ? 8 : 2}
        sx={{
          p: 4,
          border: isDragOver ? '2px dashed #3b82f6' : '2px dashed #d1d5db',
          borderRadius: 2,
          backgroundColor: isDragOver ? '#f0f9ff' : 'white',
          transition: 'all 0.3s ease',
          cursor: selectedFile ? 'default' : 'pointer',
        }}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !selectedFile && document.getElementById('file-input')?.click()}
      >
        <Box 
          display="flex" 
          flexDirection="column" 
          alignItems="center" 
          textAlign="center"
        >
          {!selectedFile ? (
            <>
              <UploadIcon 
                sx={{ 
                  fontSize: 48, 
                  color: isDragOver ? '#3b82f6' : '#9ca3af',
                  mb: 2
                }} 
              />
              <Typography variant="h6" gutterBottom>
                {isDragOver ? 'Drop your file here' : 'Upload Document'}
              </Typography>
              <Typography variant="body2" color="textSecondary" mb={2}>
                Drag and drop your file here, or click to browse
              </Typography>
              <Box display="flex" gap={1} flexWrap="wrap" justifyContent="center">
                {acceptedTypes.map((type) => (
                  <Chip 
                    key={type}
                    label={type.split('/')[1].toUpperCase()}
                    size="small"
                    variant="outlined"
                  />
                ))}
              </Box>
              <Typography variant="caption" color="textSecondary" mt={1}>
                Maximum file size: {maxSize}MB
              </Typography>
            </>
          ) : (
            <>
              <Box 
                display="flex" 
                alignItems="center" 
                gap={2} 
                mb={2}
                p={2}
                bgcolor="#f8fafc"
                borderRadius={1}
                width="100%"
              >
                <Typography fontSize="2rem">
                  {getFileIcon(selectedFile.type)}
                </Typography>
                <Box flex={1}>
                  <Typography variant="subtitle1" fontWeight={600}>
                    {selectedFile.name}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • {selectedFile.type || 'Unknown'}
                  </Typography>
                </Box>
                {uploadProgress === 0 && (
                  <IconButton 
                    onClick={handleRemoveFile}
                    color="error"
                    size="small"
                  >
                    <DeleteIcon />
                  </IconButton>
                )}
                {uploadProgress === 100 && (
                  <CheckIcon color="success" />
                )}
              </Box>

              {uploadProgress > 0 && uploadProgress < 100 && (
                <Box width="100%" mb={2}>
                  <LinearProgress 
                    variant="determinate" 
                    value={uploadProgress} 
                    sx={{ 
                      height: 8, 
                      borderRadius: 4,
                      backgroundColor: '#e5e7eb',
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: '#3b82f6'
                      }
                    }}
                  />
                  <Typography variant="body2" color="textSecondary" mt={1}>
                    Uploading... {uploadProgress}%
                  </Typography>
                </Box>
              )}

              {uploadProgress === 0 && (
                <Button
                  variant="contained"
                  onClick={handleUpload}
                  disabled={isLoading}
                  sx={{
                    mt: 2,
                    px: 4,
                    py: 1.5,
                    backgroundColor: '#3b82f6',
                    '&:hover': {
                      backgroundColor: '#2563eb'
                    }
                  }}
                >
                  {isLoading ? 'Processing...' : 'Upload & Process'}
                </Button>
              )}
            </>
          )}
        </Box>
      </Paper>

      {/* Hidden file input */}
      <input
        id="file-input"
        type="file"
        hidden
        accept={acceptedTypes.join(',')}
        onChange={handleFileInput}
      />

      {/* Info Alert */}
      <Alert severity="info" sx={{ mt: 2 }}>
        <Typography variant="body2">
          <strong>Supported formats:</strong> JPEG, PNG, TIFF images and PDF documents. 
          The system will automatically detect tables, extract data, and provide insights.
        </Typography>
      </Alert>
    </Box>
  );
};

export default FileUpload;