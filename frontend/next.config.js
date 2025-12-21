/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  output: 'standalone', // For Docker deployment
  images: {
    domains: ['localhost'],
  },
  // Remove rewrites for separate deployment
};

module.exports = nextConfig;