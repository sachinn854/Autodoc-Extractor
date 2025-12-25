import React from 'react';
import Head from 'next/head';
import { useAuth } from '../contexts/AuthContext';

interface LayoutProps {
  children: React.ReactNode;
  title?: string;
  description?: string;
}

const Layout: React.FC<LayoutProps> = ({ 
  children, 
  title = 'Restaurant Bill Analyzer',
  description = 'AI-powered restaurant bill processing and analysis'
}) => {
  const { user, logout } = useAuth();

  return (
    <>
      <Head>
        <title>{title}</title>
        <meta name="description" content={description} />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gray-50">
        {/* Navigation Header */}
        <nav className="bg-white shadow-sm border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              {/* Logo */}
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <h1 className="text-xl font-bold text-blue-600">
                    🍽️ Bill Analyzer
                  </h1>
                </div>
              </div>

              {/* Navigation Links */}
              <div className="hidden md:block">
                <div className="ml-10 flex items-baseline space-x-4">
                  {user ? (
                    <>
                      <a
                        href="/dashboard"
                        className="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium transition-colors"
                      >
                        Dashboard
                      </a>
                      <a
                        href="/upload"
                        className="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium transition-colors"
                      >
                        Upload Bill
                      </a>
                      <div className="flex items-center space-x-3">
                        <span className="text-sm text-gray-600">
                          Welcome, {user.email}
                        </span>
                        <button
                          onClick={logout}
                          className="btn btn-outline text-sm"
                        >
                          Logout
                        </button>
                      </div>
                    </>
                  ) : (
                    <>
                      <a
                        href="/login"
                        className="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium transition-colors"
                      >
                        Login
                      </a>
                      <a
                        href="/signup"
                        className="btn btn-primary text-sm"
                      >
                        Sign Up
                      </a>
                    </>
                  )}
                </div>
              </div>

              {/* Mobile menu button */}
              <div className="md:hidden">
                <button
                  type="button"
                  className="text-gray-700 hover:text-blue-600 focus:outline-none focus:text-blue-600"
                  aria-label="Toggle menu"
                >
                  <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1">
          {children}
        </main>

        {/* Footer */}
        <footer className="bg-white border-t border-gray-200 mt-auto">
          <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center">
              <div className="text-sm text-gray-600">
                © 2024 Restaurant Bill Analyzer. Built with ❤️ for food lovers.
              </div>
              <div className="flex space-x-4">
                <a
                  href="/docs"
                  className="text-sm text-gray-600 hover:text-blue-600 transition-colors"
                >
                  API Docs
                </a>
                <a
                  href="/privacy"
                  className="text-sm text-gray-600 hover:text-blue-600 transition-colors"
                >
                  Privacy
                </a>
                <a
                  href="/support"
                  className="text-sm text-gray-600 hover:text-blue-600 transition-colors"
                >
                  Support
                </a>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </>
  );
};

export default Layout;