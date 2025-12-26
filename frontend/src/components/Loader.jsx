import React from 'react';

const Loader = ({ 
  message = 'Loading...', 
  progress, 
  size = 'medium' 
}) => {
  const sizeClasses = {
    small: 'w-6 h-6',
    medium: 'w-10 h-10',
    large: 'w-16 h-16'
  };

  const textSizeClasses = {
    small: 'text-sm',
    medium: 'text-base',
    large: 'text-lg'
  };

  return (
    <div className="flex flex-col items-center justify-center p-8">
      <div className="relative inline-flex mb-4">
        {progress !== undefined ? (
          // Determinate progress
          <div className="relative">
            <svg className={`${sizeClasses[size]} transform -rotate-90`} viewBox="0 0 36 36">
              <path
                className="text-gray-200"
                stroke="currentColor"
                strokeWidth="3"
                fill="none"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              />
              <path
                className="text-blue-600"
                stroke="currentColor"
                strokeWidth="3"
                strokeDasharray={`${progress}, 100`}
                strokeLinecap="round"
                fill="none"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className={`font-semibold text-gray-700 ${size === 'small' ? 'text-xs' : 'text-sm'}`}>
                {Math.round(progress)}%
              </span>
            </div>
          </div>
        ) : (
          // Indeterminate spinner
          <div className={`${sizeClasses[size]} animate-spin`}>
            <svg className="w-full h-full text-blue-600" fill="none" viewBox="0 0 24 24">
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
          </div>
        )}
      </div>
      
      <p className={`text-gray-600 text-center font-medium max-w-xs leading-relaxed ${textSizeClasses[size]}`}>
        {message}
      </p>

      {/* Animated dots for indeterminate loading */}
      {progress === undefined && (
        <div className="mt-2">
          <p className="text-sm text-gray-500 animate-pulse">
            Please wait...
          </p>
        </div>
      )}
    </div>
  );
};

export default Loader;