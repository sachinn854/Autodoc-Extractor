// API Response Types
export interface ApiResponse<T = any> {
  status: 'success' | 'error' | 'processing' | 'completed' | 'failed';
  message?: string;
  data?: T;
}

// Document Processing Types
export interface Item {
  description: string;
  qty: number;
  unit_price: number;
  line_total: number;
  category?: string;
}

export interface ExtractedResult {
  vendor: string;
  date: string;
  items: Item[];
  total: number;
  currency?: string;
  document_type?: string;
}

export interface TableDetectionResult {
  tables_found: number;
  confidence_score: number;
  bounding_boxes?: Array<{
    x: number;
    y: number;
    width: number;
    height: number;
  }>;
}

export interface CategoryInsight {
  name: string;
  total_amount: number;
  percentage: number;
  count: number;
  color?: string;
}

export interface AnomalyData {
  item_index: number;
  anomaly_type: string;
  description: string;
  severity: 'low' | 'medium' | 'high';
  confidence: number;
}

export interface SpendingTrend {
  period: string; // YYYY-MM format
  amount: number;
  transaction_count: number;
}

export interface Insights {
  categories: CategoryInsight[];
  anomalies: AnomalyData[];
  spending_trends: SpendingTrend[];
  total_amount: number;
  average_transaction: number;
  top_vendor: string;
  summary: {
    most_expensive_category: string;
    most_frequent_category: string;
    anomaly_count: number;
  };
}

// Job Management Types
export interface JobStatus {
  job_id: string;
  status: 'uploaded' | 'processing' | 'completed' | 'failed';
  progress: string;
  created_at: string;
  completed_at?: string;
  error?: string;
}

export interface UploadResponse {
  job_id: string;
  status: 'uploaded';
  filename: string;
  process_url: string;
  message: string;
}

export interface ProcessResponse {
  job_id: string;
  status: 'processing';
  result_url: string;
  estimated_time: string;
}

export interface ResultResponse {
  job_id: string;
  status: 'completed' | 'failed';
  extracted?: ExtractedResult;
  insights?: Insights;
  tables?: TableDetectionResult;
  csv_url?: string;
  files: {
    [key: string]: string; // filename -> download_url
  };
  error?: string;
}

// UI Component Types
export interface FileUploadProps {
  onUploadComplete: (response: UploadResponse) => void;
  onUploadError: (error: string) => void;
  isLoading?: boolean;
  acceptedTypes?: string[];
  maxSize?: number; // in MB
}

export interface EditableTableProps {
  data: Item[];
  onDataChange: (updatedData: Item[]) => void;
  onSave: () => void;
  isReadOnly?: boolean;
  isLoading?: boolean;
}

export interface InsightsPanelProps {
  insights: Insights;
  isLoading?: boolean;
}

export interface LoaderProps {
  message?: string;
  progress?: number;
  size?: 'small' | 'medium' | 'large';
}

export interface ImagePreviewProps {
  imageUrl: string;
  boundingBoxes?: Array<{
    x: number;
    y: number;
    width: number;
    height: number;
  }>;
  onImageLoad?: () => void;
}

// Form Types
export interface EditFormData {
  items: Item[];
  vendor?: string;
  date?: string;
  total?: number;
}

// Chart Data Types
export interface ChartData {
  name: string;
  value: number;
  color?: string;
  percentage?: number;
}

export interface TrendChartData {
  period: string;
  amount: number;
  count: number;
}

// Error Types
export interface AppError {
  message: string;
  code?: string;
  details?: any;
}

// Navigation Types
export interface NavItem {
  label: string;
  href: string;
  icon?: string;
  active?: boolean;
}

// Theme Types
export interface ThemeConfig {
  primary: string;
  secondary: string;
  success: string;
  warning: string;
  error: string;
  background: string;
}

// Utility Types
export type ProcessingStage = 
  | 'upload'
  | 'preprocessing' 
  | 'ocr'
  | 'table_detection'
  | 'parsing'
  | 'ml_categorization'
  | 'insights_generation'
  | 'completed';

export type FileType = 'image/jpeg' | 'image/png' | 'image/tiff' | 'application/pdf';

export type SortDirection = 'asc' | 'desc';

export interface SortConfig {
  key: keyof Item;
  direction: SortDirection;
}