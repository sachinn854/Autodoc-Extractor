import React, { useState } from 'react';
import { useRouter } from 'next/router';
import Layout from '../components/Layout';
import FileUpload from '../components/FileUpload';
import { useAuth } from '../contexts/AuthContext';

const UploadPage = () => {
  const router = useRouter();
  const { user } = useAuth();
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleUploadComplete = (response) => {
    setSuccess('File uploaded successfully! Processing your bill...');
    setError('');
    
    // Redirect to results page after a short delay
    setTimeout(() => {
      router.push(`/result/${response.job_id}`);
    }, 2000);
  };

  const handleUploadError = (errorMessage) => {
    setError(errorMessage);
    setSuccess('');
    setIsLoading(false);
  };

  // Redirect to login if not authenticated
  if (!user) {
    return (
      <Layout title="Upload - Restaurant Bill Analyzer">
        <div className="min-h-screen flex items-center justify-center py-12 px-4">
          <div className="max-w-md w-full text-center">
            <div className="text-6xl mb-4">🔒</div>
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              Login Required
            </h2>
            <p className="text-gray-600 mb-6">
              Please log in to upload and process your restaurant bills.
            </p>
            <div className="space-y-3">
              <button
                onClick={() => router.push('/login')}
                className="w-full btn btn-primary"
              >
                Login to Continue
              </button>
              <button
                onClick={() => router.push('/signup')}
                className="w-full btn btn-outline"
              >
                Create New Account
              </button>
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout title="Upload Bill - Restaurant Bill Analyzer">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Upload Your Restaurant Bill
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Upload a photo or scan of your restaurant bill and let our AI extract all the details automatically.
          </p>
        </div>

        {/* Success/Error Messages */}
        {success && (
          <div className="alert alert-success mb-6 flex items-center">
            <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            {success}
          </div>
        )}

        {error && (
          <div className="alert alert-error mb-6 flex items-center">
            <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            {error}
          </div>
        )}

        {/* File Upload Component */}
        <FileUpload
          onUploadComplete={handleUploadComplete}
          onUploadError={handleUploadError}
          isLoading={isLoading}
        />

        {/* Features Section */}
        <div className="mt-16">
          <h2 className="text-2xl font-bold text-center text-gray-900 mb-8">
            What Our AI Extracts
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-4xl mb-3">🏪</div>
              <h3 className="font-semibold mb-2">Restaurant Info</h3>
              <p className="text-sm text-gray-600">Name, address, phone number</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-3">🍽️</div>
              <h3 className="font-semibold mb-2">Menu Items</h3>
              <p className="text-sm text-gray-600">Food items, quantities, prices</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-3">💰</div>
              <h3 className="font-semibold mb-2">Financial Details</h3>
              <p className="text-sm text-gray-600">Subtotal, tax, tip, total</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-3">📊</div>
              <h3 className="font-semibold mb-2">Insights</h3>
              <p className="text-sm text-gray-600">Spending analysis, trends</p>
            </div>
          </div>
        </div>

        {/* Tips Section */}
        <div className="mt-12 card">
          <h3 className="text-lg font-semibold mb-4">📸 Tips for Best Results</h3>
          <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-600">
            <div>
              <h4 className="font-medium text-gray-900 mb-2">Photo Quality</h4>
              <ul className="space-y-1">
                <li>• Ensure good lighting</li>
                <li>• Keep the bill flat and straight</li>
                <li>• Avoid shadows and glare</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-gray-900 mb-2">File Format</h4>
              <ul className="space-y-1">
                <li>• JPEG, PNG, or PDF files</li>
                <li>• Maximum 10MB file size</li>
                <li>• High resolution preferred</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default UploadPage;