import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Alert,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  RestoreFromTrash as ResetIcon
} from '@mui/icons-material';
import { ImagePreviewProps } from '../types/schema';

const ImagePreview: React.FC<ImagePreviewProps> = ({ 
  imageUrl, 
  boundingBoxes = []
}) => {
  const [zoom, setZoom] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.2, 3));
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.2, 0.5));
  const handleReset = () => {
    setZoom(1);
    setPosition({ x: 0, y: 0 });
  };



  if (!imageUrl) {
    return (
      <Alert severity="info">
        <Typography>No image preview available.</Typography>
      </Alert>
    );
  }

  return (
    <Box className="w-full">
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6" fontWeight={600}>
          🖼️ Document Preview
        </Typography>
        
        <Box display="flex" gap={1}>
          <Tooltip title="Zoom In">
            <IconButton onClick={handleZoomIn} size="small">
              <ZoomInIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Zoom Out">
            <IconButton onClick={handleZoomOut} size="small">
              <ZoomOutIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Reset View">
            <IconButton onClick={handleReset} size="small">
              <ResetIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      <Paper elevation={2} sx={{ p: 2, overflow: 'hidden', height: 600 }}>
        <Box 
          sx={{ 
            width: '100%', 
            height: '100%', 
            overflow: 'auto',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center'
          }}
        >
          <Box
            sx={{
              transform: `scale(${zoom}) translate(${position.x}px, ${position.y}px)`,
              transition: 'transform 0.2s ease',
              position: 'relative'
            }}
          >
            <img
              src={imageUrl}
              alt="Document Preview"
              style={{
                maxWidth: '100%',
                maxHeight: '100%',
                objectFit: 'contain'
              }}
            />
            
            {/* Bounding boxes overlay */}
            {boundingBoxes.length > 0 && (
              <svg
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  pointerEvents: 'none'
                }}
              >
                {boundingBoxes.map((box, index) => (
                  <rect
                    key={index}
                    x={box.x}
                    y={box.y}
                    width={box.width}
                    height={box.height}
                    fill="transparent"
                    stroke="#3b82f6"
                    strokeWidth="2"
                    strokeDasharray="5,5"
                  />
                ))}
              </svg>
            )}
          </Box>
        </Box>
      </Paper>

      <Box mt={2}>
        <Alert severity="info">
          <Typography variant="body2">
            <strong>Controls:</strong> Use zoom buttons to adjust view. 
            {boundingBoxes.length > 0 && ` ${boundingBoxes.length} table(s) detected and highlighted.`}
          </Typography>
        </Alert>
      </Box>
    </Box>
  );
};

export default ImagePreview;