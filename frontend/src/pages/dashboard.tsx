import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import Layout from '../components/Layout';
import { useAuth } from '../contexts/AuthContext';

// API URL configuration
const API_URL = process.env.NEXT_PUBLIC_API_URL || 
  (process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8001');

interface Document {
  id: number;
  job_id: string;
  filename: string;
  status: string;
  created_at: string;
  updated_at: string;
  items_count?: number;
  vendor?: string;
  total_amount?: string;
}

const Dashboard: React.FC = () => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const { user, token, logout, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && !token) {
      router.push('/login');
      return;
    }

    if (token) {
      fetchDocuments();
    }
  }, [token, isLoading, router]);

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_URL}/my-documents`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        throw new Error('Failed to fetch documents');
      }

      const data = await response.json();
      setDocuments(data.documents);
    } catch (err: any) {
      setError(err.message || 'Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'processing':
        return 'bg-yellow-100 text-yellow-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString);
      // Check if date is valid
      if (isNaN(date.getTime())) {
        return 'Invalid Date';
      }
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        hour12: true
      });
    } catch (error) {
      console.error('Date formatting error:', error);
      return 'Invalid Date';
    }
  };

  if (isLoading || loading) {
    return (
      <Layout title="Dashboard - Restaurant Bill Analyzer">
        <div className="min-h-screen flex items-center justify-center">
          <div className="text-center">
            <div className="spinner w-8 h-8 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading your documents...</p>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout title="Dashboard - Restaurant Bill Analyzer">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">My Documents</h1>
            <p className="text-gray-600">
              Manage and view your processed restaurant bills
            </p>
          </div>
          <button
            onClick={() => router.push('/upload')}
            className="btn btn-primary mt-4 sm:mt-0 flex items-center"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            Upload New Bill
          </button>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="alert alert-error mb-6 flex items-center">
            <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            {error}
          </div>
        )}

        {/* Documents Table */}
        <div className="card">
          {documents.length === 0 ? (
            <div className="text-center py-12">
              <div className="text-6xl mb-4">📄</div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                No documents yet
              </h3>
              <p className="text-gray-600 mb-6">
                Upload your first restaurant bill to get started with AI-powered processing
              </p>
              <button
                onClick={() => router.push('/upload')}
                className="btn btn-primary"
              >
                Upload Your First Bill
              </button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4 font-semibold text-gray-900">File Name</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-900">Restaurant</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-900">Status</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-900">Items</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-900">Total</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-900">Uploaded</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-900">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {documents.map((doc) => (
                    <tr key={doc.id} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-4 px-4">
                        <div className="flex items-center">
                          <div className="text-2xl mr-3">🧾</div>
                          <div>
                            <div className="font-medium text-gray-900">{doc.filename}</div>
                            <div className="text-sm text-gray-500">ID: {doc.job_id.slice(0, 8)}...</div>
                          </div>
                        </div>
                      </td>
                      <td className="py-4 px-4 text-gray-900">
                        {doc.vendor || '-'}
                      </td>
                      <td className="py-4 px-4">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(doc.status)}`}>
                          {doc.status}
                        </span>
                      </td>
                      <td className="py-4 px-4 text-gray-900">
                        {doc.items_count !== undefined ? doc.items_count : '-'}
                      </td>
                      <td className="py-4 px-4 text-gray-900 font-medium">
                        {doc.total_amount || '-'}
                      </td>
                      <td className="py-4 px-4 text-gray-600 text-sm">
                        {formatDate(doc.created_at)}
                      </td>
                      <td className="py-4 px-4">
                        {doc.status === 'completed' && (
                          <button
                            onClick={() => router.push(`/result/${doc.job_id}`)}
                            className="btn btn-outline text-sm"
                          >
                            View Results
                          </button>
                        )}
                        {doc.status === 'processing' && (
                          <div className="flex items-center text-yellow-600">
                            <div className="spinner w-4 h-4 mr-2"></div>
                            Processing...
                          </div>
                        )}
                        {doc.status === 'failed' && (
                          <span className="text-red-600 text-sm">Failed</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Stats Cards */}
        {documents.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
            <div className="card text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">
                {documents.length}
              </div>
              <div className="text-gray-600">Total Documents</div>
            </div>
            <div className="card text-center">
              <div className="text-3xl font-bold text-green-600 mb-2">
                {documents.filter(d => d.status === 'completed').length}
              </div>
              <div className="text-gray-600">Processed</div>
            </div>
            <div className="card text-center">
              <div className="text-3xl font-bold text-yellow-600 mb-2">
                {documents.filter(d => d.status === 'processing').length}
              </div>
              <div className="text-gray-600">Processing</div>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
};
export default Dashboard;