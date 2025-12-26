import React, { useState, useCallback } from 'react';
import apiService from '../services/api';
import { useAuth } from '../contexts/AuthContext';

const FileUpload = ({
  onUploadComplete,
  onUploadError,
  isLoading = false,
  acceptedTypes = ['image/jpeg', 'image/png', 'image/tiff', 'application/pdf'],
  maxSize = 10 // 10MB default
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const { token } = useAuth();

  // Drag and drop handlers
  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelection(files[0]);
    }
  }, []);

  // File selection validation
  const handleFileSelection = (file) => {
    // Validate file type
    if (!acceptedTypes.includes(file.type)) {
      onUploadError(`Unsupported file type: ${file.type}. Please upload ${acceptedTypes.join(', ')}`);
      return;
    }

    // Validate file size (convert MB to bytes)
    const maxSizeBytes = maxSize * 1024 * 1024;
    if (file.size > maxSizeBytes) {
      onUploadError(`File too large: ${(file.size / (1024 * 1024)).toFixed(2)}MB. Maximum size: ${maxSize}MB`);
      return;
    }

    setSelectedFile(file);
    setUploadProgress(0);
  };

  const handleFileInput = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelection(file);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setUploadProgress(0);
    // Reset file input
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
      fileInput.value = '';
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    try {
      setUploadProgress(10);
      
      // Upload file using API service
      const response = await apiService.uploadFile(selectedFile);
      
      setUploadProgress(100);
      
      // Notify parent component
      setTimeout(() => {
        onUploadComplete(response);
      }, 500);

    } catch (error) {
      setUploadProgress(0);
      onUploadError(error.message || 'Upload failed');
    }
  };

  const getFileIcon = (fileType) => {
    if (fileType.startsWith('image/')) {
      return '🖼️';
    }
    if (fileType === 'application/pdf') {
      return '📄';
    }
    return '📎';
  };

  const formatFileSize = (bytes) => {
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Upload Area */}
      <div
        className={`
          file-upload-area cursor-pointer transition-all duration-300
          ${isDragOver ? 'dragover border-blue-500 bg-blue-50' : 'border-gray-300'}
          ${selectedFile ? 'cursor-default' : ''}
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !selectedFile && document.getElementById('file-input')?.click()}
      >
        {!selectedFile ? (
          <div className="text-center">
            <div className="text-6xl mb-4">
              {isDragOver ? '📤' : '📁'}
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">
              {isDragOver ? 'Drop your file here' : 'Upload Restaurant Bill'}
            </h3>
            <p className="text-gray-600 mb-4">
              Drag and drop your file here, or click to browse
            </p>
            <div className="flex flex-wrap gap-2 justify-center mb-4">
              {acceptedTypes.map((type) => (
                <span 
                  key={type}
                  className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full"
                >
                  {type.split('/')[1].toUpperCase()}
                </span>
              ))}
            </div>
            <p className="text-sm text-gray-500">
              Maximum file size: {maxSize}MB
            </p>
          </div>
        ) : (
          <div>
            <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg mb-4">
              <div className="text-3xl">
                {getFileIcon(selectedFile.type)}
              </div>
              <div className="flex-1">
                <div className="font-semibold text-gray-900">{selectedFile.name}</div>
                <div className="text-sm text-gray-600">
                  {formatFileSize(selectedFile.size)} • {selectedFile.type || 'Unknown'}
                </div>
              </div>
              {uploadProgress === 0 && (
                <button
                  onClick={handleRemoveFile}
                  className="text-red-600 hover:text-red-700 p-1"
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </button>
              )}
              {uploadProgress === 100 && (
                <div className="text-green-600">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                </div>
              )}
            </div>

            {uploadProgress > 0 && uploadProgress < 100 && (
              <div className="mb-4">
                <div className="progress-bar">
                  <div 
                    className="progress-fill"
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
                <p className="text-sm text-gray-600 mt-2 text-center">
                  Uploading... {uploadProgress}%
                </p>
              </div>
            )}

            {uploadProgress === 0 && (
              <div className="text-center">
                <button
                  onClick={handleUpload}
                  disabled={isLoading}
                  className="btn btn-primary px-8 py-3"
                >
                  {isLoading ? (
                    <div className="flex items-center">
                      <div className="spinner mr-2"></div>
                      Processing...
                    </div>
                  ) : (
                    'Upload & Process'
                  )}
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Hidden file input */}
      <input
        id="file-input"
        type="file"
        accept={acceptedTypes.join(',')}
        onChange={handleFileInput}
        className="hidden"
      />

      {/* Info Alert */}
      <div className="alert alert-info mt-4">
        <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
        </svg>
        <div>
          <strong>Supported formats:</strong> JPEG, PNG, TIFF images and PDF documents. 
          The AI will automatically detect tables, extract menu items, prices, and provide spending insights.
        </div>
      </div>
    </div>
  );
};

export default FileUpload;