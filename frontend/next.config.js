/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  output: 'export', // Static export for FastAPI serving
  trailingSlash: true,
  images: {
    unoptimized: true, // Required for static export
  },
  // API calls will go to same origin (FastAPI backend)
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: '/api/:path*', // Same origin API calls
      },
    ];
  },
};

module.exports = nextConfig;