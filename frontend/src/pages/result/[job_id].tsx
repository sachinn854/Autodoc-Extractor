import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Layout from '../../components/Layout';
import EditableTable from '../../components/EditableTable';
import Loader from '../../components/Loader';
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
        <div className="py-6">
          {children}
        </div>
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
      setTimeout(() => setShowSuccess(false), 3000);
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
        title: 'Restaurant Bill Analyzer Results',
        text: 'Check out my bill processing results',
        url: url
      });
    } else {
      navigator.clipboard.writeText(url);
      setShowSuccess(true);
      setTimeout(() => setShowSuccess(false), 3000);
    }
  };

  if (loading) {
    return (
      <Layout title="Processing - Restaurant Bill Analyzer">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="card text-center">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              Processing Your Bill
            </h2>
            {jobStatus && (
              <p className="text-gray-600 mb-6">
                {jobStatus.progress}
              </p>
            )}
            <Loader
              message={jobStatus?.progress || 'Processing your document...'}
              size="large"
            />
            {jobStatus?.status === 'processing' && (
              <div className="alert alert-info mt-6 max-w-md mx-auto">
                {jobStatus.progress?.includes('Downloading OCR models') ? (
                  <>
                    First time setup: Downloading AI models (2-3 minutes).
                    <br />Subsequent uploads will be much faster.
                  </>
                ) : (
                  'This usually takes 30-60 seconds depending on document complexity.'
                )}
              </div>
            )}
          </div>
        </div>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout title="Error - Restaurant Bill Analyzer">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="alert alert-error mb-6">
            <svg className="w-6 h-6 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <div>
              <h3 className="font-semibold">Processing Failed</h3>
              <p>{error}</p>
            </div>
          </div>
          <button
            onClick={() => router.push('/upload')}
            className="btn btn-primary flex items-center"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Try Again
          </button>
        </div>
      </Layout>
    );
  }

  if (!results) {
    return (
      <Layout title="Results Not Found - Restaurant Bill Analyzer">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="alert alert-warning">
            <svg className="w-6 h-6 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <div>
              <h3 className="font-semibold">Results Not Found</h3>
              <p>The requested job results could not be found.</p>
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout title="Results - Restaurant Bill Analyzer">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8">
          <div className="flex items-center gap-4">
            <button
              onClick={() => router.push('/dashboard')}
              className="p-2 text-gray-600 hover:text-blue-600 transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
            </button>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Processing Results
              </h1>
              <div className="flex items-center gap-3 mt-2">
                <p className="text-sm text-gray-600">
                  Job ID: {job_id}
                </p>
                {jobStatus && (
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${jobStatus.status === 'completed' ? 'bg-green-100 text-green-800' :
                    jobStatus.status === 'processing' ? 'bg-yellow-100 text-yellow-800' :
                      jobStatus.status === 'failed' ? 'bg-red-100 text-red-800' :
                        'bg-gray-100 text-gray-800'
                    }`}>
                    {jobStatus.status === 'processing' && (
                      <div className="inline-block w-3 h-3 mr-1">
                        <div className="spinner w-3 h-3"></div>
                      </div>
                    )}
                    {jobStatus.status}
                  </span>
                )}
                {dataModified && (
                  <span className="px-2 py-1 rounded-full text-xs font-medium bg-orange-100 text-orange-800">
                    Modified
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 mt-4 sm:mt-0">
            <button
              onClick={() => window.location.reload()}
              className="p-2 text-gray-600 hover:text-blue-600 transition-colors"
              title="Refresh Results"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>

            <button
              onClick={handleShare}
              className="p-2 text-gray-600 hover:text-blue-600 transition-colors"
              title="Share Results"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.367 2.684 3 3 0 00-5.367-2.684z" />
              </svg>
            </button>

            {results.csv_url && (
              <button
                onClick={handleDownloadCSV}
                className="btn btn-outline text-sm flex items-center"
              >
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Download CSV
              </button>
            )}

            {dataModified && (
              <button
                onClick={handleSaveChanges}
                disabled={saving}
                className="btn btn-primary text-sm"
              >
                {saving ? (
                  <div className="flex items-center">
                    <div className="spinner w-4 h-4 mr-2"></div>
                    Saving...
                  </div>
                ) : (
                  'Save Changes'
                )}
              </button>
            )}
          </div>
        </div>

        {/* Summary Stats */}
        {results?.extracted && (
          <div className="card mb-8 bg-gray-50">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 text-center">
              <div>
                <div className="text-3xl font-bold text-blue-600 mb-1">
                  {results.extracted.items.length}
                </div>
                <div className="text-sm text-gray-600">Items Extracted</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-green-600 mb-1">
                  ${extractedData.reduce((sum, item) => sum + (Number(item.line_total) || 0), 0).toFixed(2)}
                </div>
                <div className="text-sm text-gray-600">Total Amount</div>
              </div>
              <div>
                <div className="text-xl font-semibold text-gray-900 mb-1">
                  {results.extracted.vendor || 'Unknown'}
                </div>
                <div className="text-sm text-gray-600">Restaurant</div>
              </div>
              <div>
                <div className="text-xl font-semibold text-gray-900 mb-1">
                  {results.extracted.date || 'Unknown'}
                </div>
                <div className="text-sm text-gray-600">Date</div>
              </div>
            </div>
          </div>
        )}

        {/* Tabs */}
        <div className="card">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8">
              <button
                onClick={() => setActiveTab(0)}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 0
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
              >
                📊 Data Table
              </button>
              <button
                onClick={() => setActiveTab(1)}
                disabled={!results.insights}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 1
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  } ${!results.insights ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                📈 Insights
              </button>
              <button
                onClick={() => setActiveTab(2)}
                disabled={!results.files?.['processed_image.jpg']}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 2
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  } ${!results.files?.['processed_image.jpg'] ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                🖼️ Preview
              </button>
            </nav>
          </div>

          <TabPanel value={activeTab} index={0}>
            {results?.extracted ? (
              <EditableTable
                data={extractedData}
                onDataChange={handleDataChange}
                onSave={handleSaveChanges}
                isLoading={saving}
              />
            ) : (
              <div className="alert alert-info">
                <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
                <div>
                  <p className="font-medium">No data was extracted from the document.</p>
                  <p className="text-sm mt-1">This could mean:</p>
                  <ul className="text-sm mt-2 list-disc list-inside">
                    <li>No tables were detected in the document</li>
                    <li>The image quality was too low for accurate OCR</li>
                    <li>The document format is not supported</li>
                  </ul>
                </div>
              </div>
            )}
          </TabPanel>

          <TabPanel value={activeTab} index={1}>
            <div className="alert alert-info">
              <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
              <div>
                <h3 className="font-semibold mb-2">📈 Insights Panel</h3>
                <p>Insights feature temporarily disabled for debugging.</p>
                {results.insights && (
                  <p className="text-sm mt-2">
                    Raw insights data available: {JSON.stringify(results.insights).substring(0, 100)}...
                  </p>
                )}
              </div>
            </div>
          </TabPanel>

          <TabPanel value={activeTab} index={2}>
            <div>
              <h3 className="text-lg font-semibold mb-4">🖼️ Image Preview</h3>
              {results.files?.['processed_image.jpg'] ? (
                <div className="border border-gray-200 rounded-lg p-4">
                  <img
                    src={results.files['processed_image.jpg']}
                    alt="Document Preview"
                    className="max-w-full max-h-96 object-contain mx-auto"
                  />
                  <p className="text-sm text-gray-600 mt-2 text-center">
                    Document processed successfully
                  </p>
                </div>
              ) : (
                <div className="alert alert-info">
                  <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                  </svg>
                  Image preview is not available. The processed image may not have been saved.
                </div>
              )}
            </div>
          </TabPanel>
        </div>

        {/* Success Notification */}
        {showSuccess && (
          <div className="fixed bottom-4 right-4 z-50">
            <div className="alert alert-success flex items-center shadow-lg">
              <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              Changes saved successfully!
              <button
                onClick={() => setShowSuccess(false)}
                className="ml-4 text-green-800 hover:text-green-900"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </button>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
};

export default ResultPage;