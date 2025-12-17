import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Alert
} from '@mui/material';
import { InsightsPanelProps } from '../types/schema';

const InsightsPanel: React.FC<InsightsPanelProps> = ({ insights, isLoading = false }) => {
  
  if (isLoading) {
    return (
      <Box p={4} textAlign="center">
        <Typography>Loading insights...</Typography>
      </Box>
    );
  }

  return (
    <Box className="w-full">
      <Typography variant="h5" fontWeight={600} mb={3}>
        📊 Spending Insights
      </Typography>

      <Paper elevation={2} sx={{ p: 3 }}>
        <Typography variant="h6" mb={2}>
          Analysis Summary
        </Typography>
        
        <Alert severity="info">
          <Typography variant="body2">
            Total Amount: <strong>${insights.total_amount?.toFixed(2) || '0.00'}</strong>
            <br />
            Average Transaction: <strong>${insights.average_transaction?.toFixed(2) || '0.00'}</strong>
            <br />
            Top Vendor: <strong>{insights.top_vendor || 'Unknown'}</strong>
          </Typography>
        </Alert>

        {insights.categories && insights.categories.length > 0 && (
          <Box mt={3}>
            <Typography variant="h6" mb={2}>
              Categories
            </Typography>
            {insights.categories.map((category, index) => (
              <Box key={index} mb={1}>
                <Typography variant="body1">
                  {category.name}: ${category.total_amount?.toFixed(2)} ({category.percentage?.toFixed(1)}%)
                </Typography>
              </Box>
            ))}
          </Box>
        )}

        {insights.anomalies && insights.anomalies.length > 0 && (
          <Box mt={3}>
            <Typography variant="h6" mb={2}>
              Anomalies Detected: {insights.anomalies.length}
            </Typography>
            {insights.anomalies.map((anomaly, index) => (
              <Alert key={index} severity="warning" sx={{ mb: 1 }}>
                <Typography variant="body2">
                  <strong>{anomaly.anomaly_type}</strong>: {anomaly.description}
                </Typography>
              </Alert>
            ))}
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default InsightsPanel;