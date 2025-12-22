import axios, { AxiosResponse } from 'axios';
import {
  UploadResponse,
  ProcessResponse,
  ResultResponse,
  JobStatus,
  ExtractedResult,
  ApiResponse
} from '../types/schema';

// API Configuration
// Production: Use backend API URL from environment
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes for heavy AI processing
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging and auth token
api.interceptors.request.use(
  (config) => {
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);

    // Add auth token if available
    const token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null;
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    return config;
  },
  (error) => {
    console.error('❌ API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// API Service Class
class ApiService {
  // Health Check
  async healthCheck(): Promise<{ status: string; service: string }> {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      throw new Error('Backend service unavailable');
    }
  }

  // File Upload
  async uploadFile(file: File): Promise<UploadResponse> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await api.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // 1 minute for file upload
      });

      return response.data;
    } catch (error: any) {
      if (error.response?.status === 413) {
        throw new Error('File too large. Please upload a smaller file.');
      }
      if (error.response?.status === 415) {
        throw new Error('Unsupported file type. Please upload an image or PDF.');
      }
      throw new Error(error.response?.data?.message || 'Upload failed');
    }
  }

  // Start Processing
  async startProcessing(jobId: string): Promise<ProcessResponse> {
    try {
      const response = await api.post(`/process/${jobId}`);
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 404) {
        throw new Error('Job not found. Please upload the file again.');
      }
      throw new Error(error.response?.data?.message || 'Processing failed to start');
    }
  }

  // Check Job Status
  async getJobStatus(jobId: string): Promise<JobStatus> {
    try {
      const response = await api.get(`/status/${jobId}`);
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 404) {
        throw new Error('Job not found');
      }
      throw new Error(error.response?.data?.message || 'Failed to get status');
    }
  }

  // Get Processing Results
  async getResults(jobId: string): Promise<ResultResponse> {
    try {
      const response = await api.get(`/result/${jobId}`);
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 404) {
        throw new Error('Results not found');
      }
      if (error.response?.status === 202) {
        throw new Error('Processing still in progress');
      }
      throw new Error(error.response?.data?.message || 'Failed to get results');
    }
  }

  // Update Extracted Data (User Corrections)
  async updateExtractedData(jobId: string, correctedData: ExtractedResult): Promise<ApiResponse> {
    try {
      const response = await api.put(`/result/${jobId}`, correctedData);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to update data');
    }
  }

  // Download CSV
  async downloadCSV(jobId: string): Promise<Blob> {
    try {
      const response = await api.get(`/download/${jobId}.csv`, {
        responseType: 'blob',
      });
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 404) {
        throw new Error('CSV file not found');
      }
      throw new Error('Failed to download CSV');
    }
  }

  // Download File (JSON, images, etc.)
  async downloadFile(jobId: string, filename: string): Promise<Blob> {
    try {
      const response = await api.get(`/download/${jobId}/${filename}`, {
        responseType: 'blob',
      });
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 404) {
        throw new Error(`File ${filename} not found`);
      }
      throw new Error(`Failed to download ${filename}`);
    }
  }

  // Get All Jobs
  async getAllJobs(): Promise<{ active_jobs: number; jobs: JobStatus[] }> {
    try {
      const response = await api.get('/jobs');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to get jobs');
    }
  }

  // Cleanup Job
  async cleanupJob(jobId: string): Promise<ApiResponse> {
    try {
      const response = await api.delete(`/cleanup/${jobId}`);
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 404) {
        throw new Error('Job not found');
      }
      throw new Error(error.response?.data?.message || 'Failed to cleanup job');
    }
  }

  // Authentication Methods
  async signup(email: string, password: string, fullName?: string): Promise<{ access_token: string; user: any }> {
    try {
      const response = await api.post('/auth/signup', {
        email,
        password,
        full_name: fullName
      });
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 400) {
        throw new Error(error.response.data.detail || 'Email already registered');
      }
      throw new Error(error.response?.data?.detail || 'Signup failed');
    }
  }

  async login(email: string, password: string): Promise<{ access_token: string; user: any }> {
    try {
      const response = await api.post('/auth/login', {
        email,
        password
      });
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 401) {
        throw new Error('Invalid email or password');
      }
      if (error.response?.status === 403) {
        throw new Error('Email not verified. Please check your email.');
      }
      throw new Error(error.response?.data?.detail || 'Login failed');
    }
  }

  async getCurrentUser(): Promise<any> {
    try {
      const response = await api.get('/auth/me');
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 401) {
        throw new Error('Authentication required');
      }
      throw new Error(error.response?.data?.detail || 'Failed to get user info');
    }
  }

  async verifyOTP(email: string, otpCode: string): Promise<{ access_token: string; user: any }> {
    try {
      const response = await api.post('/auth/verify-otp', {
        email,
        otp_code: otpCode
      });
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 400) {
        throw new Error(error.response.data.detail || 'Invalid OTP code');
      }
      if (error.response?.status === 429) {
        throw new Error('Too many failed attempts. Please request a new OTP.');
      }
      throw new Error(error.response?.data?.detail || 'OTP verification failed');
    }
  }

  async resendOTP(email: string): Promise<{ message: string; otp_sent: boolean }> {
    try {
      const response = await api.post('/auth/resend-otp', null, {
        params: { email }
      });
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 404) {
        throw new Error('User not found');
      }
      if (error.response?.status === 400) {
        throw new Error('Email already verified');
      }
      throw new Error(error.response?.data?.detail || 'Failed to resend OTP');
    }
  }

  async getMyDocuments(skip: number = 0, limit: number = 50): Promise<{ documents: any[]; total: number }> {
    try {
      const response = await api.get(`/my-documents?skip=${skip}&limit=${limit}`);
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 401) {
        throw new Error('Authentication required');
      }
      throw new Error(error.response?.data?.detail || 'Failed to get documents');
    }
  }
}

// Utility Functions
export const apiUtils = {
  // Poll job status until completion
  async pollJobStatus(
    jobId: string,
    onProgress?: (status: JobStatus) => void,
    maxAttempts: number = 60, // 5 minutes with 5-second intervals
    interval: number = 5000
  ): Promise<JobStatus> {
    const apiService = new ApiService();

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        const status = await apiService.getJobStatus(jobId);

        if (onProgress) {
          onProgress(status);
        }

        if (status.status === 'completed' || status.status === 'failed') {
          return status;
        }

        // Wait before next check
        await new Promise(resolve => setTimeout(resolve, interval));
      } catch (error) {
        console.error(`Status check attempt ${attempt + 1} failed:`, error);

        // If it's the last attempt, throw the error
        if (attempt === maxAttempts - 1) {
          throw error;
        }

        // Wait before retry
        await new Promise(resolve => setTimeout(resolve, interval));
      }
    }

    throw new Error('Polling timeout: Job did not complete within expected time');
  },

  // Complete upload and processing flow
  async uploadAndProcess(
    file: File,
    onProgress?: (stage: string, status?: JobStatus) => void
  ): Promise<ResultResponse> {
    const apiService = new ApiService();

    try {
      // Step 1: Upload file
      onProgress?.('Uploading file...');
      const uploadResponse = await apiService.uploadFile(file);

      // Step 2: Start processing
      onProgress?.('Starting processing...');
      await apiService.startProcessing(uploadResponse.job_id);

      // Step 3: Poll for completion
      onProgress?.('Processing document...');
      const finalStatus = await this.pollJobStatus(
        uploadResponse.job_id,
        (status) => onProgress?.(status.progress, status)
      );

      if (finalStatus.status === 'failed') {
        throw new Error(finalStatus.error || 'Processing failed');
      }

      // Step 4: Get results
      onProgress?.('Retrieving results...');
      const results = await apiService.getResults(uploadResponse.job_id);

      return results;
    } catch (error) {
      console.error('Upload and process flow failed:', error);
      throw error;
    }
  },

  // Format file size for display
  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  },

  // Validate file type
  isValidFileType(file: File): boolean {
    const validTypes = [
      'image/jpeg',
      'image/png',
      'image/tiff',
      'application/pdf'
    ];
    return validTypes.includes(file.type);
  },

  // Get file type display name
  getFileTypeDisplay(file: File): string {
    const typeMap: { [key: string]: string } = {
      'image/jpeg': 'JPEG Image',
      'image/png': 'PNG Image',
      'image/tiff': 'TIFF Image',
      'application/pdf': 'PDF Document',
    };
    return typeMap[file.type] || 'Unknown File Type';
  }
};

// Export singleton instance
const apiService = new ApiService();
export default apiService;