import React from 'react';
import { CircularProgress, Box, Typography } from '@mui/material';
import { LoaderProps } from '../types/schema';

const Loader: React.FC<LoaderProps> = ({ 
  message = 'Loading...', 
  progress, 
  size = 'medium' 
}) => {
  const sizeMap = {
    small: 24,
    medium: 40,
    large: 60
  };

  const circularSize = sizeMap[size];

  return (
    <Box 
      display="flex" 
      flexDirection="column" 
      alignItems="center" 
      justifyContent="center" 
      p={4}
      className="animate-fade-in"
    >
      <Box position="relative" display="inline-flex" mb={2}>
        <CircularProgress 
          size={circularSize}
          variant={progress !== undefined ? 'determinate' : 'indeterminate'}
          value={progress}
          sx={{ 
            color: '#3b82f6',
            '& .MuiCircularProgress-circle': {
              strokeLinecap: 'round',
            }
          }}
        />
        {progress !== undefined && (
          <Box
            position="absolute"
            top={0}
            left={0}
            bottom={0}
            right={0}
            display="flex"
            alignItems="center"
            justifyContent="center"
          >
            <Typography 
              variant="caption" 
              component="div" 
              color="textSecondary"
              fontSize={size === 'small' ? '0.625rem' : '0.75rem'}
              fontWeight={600}
            >
              {`${Math.round(progress)}%`}
            </Typography>
          </Box>
        )}
      </Box>
      
      <Typography 
        variant={size === 'large' ? 'h6' : 'body1'} 
        color="textSecondary" 
        textAlign="center"
        sx={{ 
          fontWeight: 500,
          maxWidth: '280px',
          lineHeight: 1.5
        }}
      >
        {message}
      </Typography>

      {/* Animated dots for indeterminate loading */}
      {progress === undefined && (
        <Box mt={1}>
          <Typography 
            variant="body2" 
            color="textSecondary" 
            className="animate-pulse"
          >
            Please wait...
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default Loader;