import React, { useState, useCallback, useEffect } from 'react';

const EditableTable = ({
  data,
  onDataChange,
  onSave,
  isReadOnly = false,
  isLoading = false
}) => {
  const [rows, setRows] = useState([]);
  const [editingRow, setEditingRow] = useState(null);
  const [isModified, setIsModified] = useState(false);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [newItem, setNewItem] = useState({
    description: '',
    qty: 1,
    unit_price: 0,
    line_total: 0
  });

  useEffect(() => {
    setRows([...data]);
  }, [data]);

  const handleEdit = (index) => {
    setEditingRow(index);
  };

  const handleSave = (index) => {
    setEditingRow(null);
    setIsModified(true);
    onDataChange(rows);
  };

  const handleCancel = () => {
    setEditingRow(null);
    setRows([...data]); // Reset to original data
  };

  const handleDelete = (index) => {
    const updatedRows = rows.filter((_, i) => i !== index);
    setRows(updatedRows);
    setIsModified(true);
    onDataChange(updatedRows);
  };

  const handleFieldChange = (index, field, value) => {
    const updatedRows = [...rows];
    updatedRows[index] = { ...updatedRows[index], [field]: value };
    
    // Auto-calculate line total
    if (field === 'qty' || field === 'unit_price') {
      updatedRows[index].line_total = updatedRows[index].qty * updatedRows[index].unit_price;
    }
    
    setRows(updatedRows);
  };

  const handleAddItem = () => {
    if (!newItem.description?.trim()) return;
    
    const item = {
      description: newItem.description,
      qty: newItem.qty || 1,
      unit_price: newItem.unit_price || 0,
      line_total: (newItem.qty || 1) * (newItem.unit_price || 0),
      category: newItem.category
    };
    
    const updatedRows = [...rows, item];
    setRows(updatedRows);
    setIsModified(true);
    onDataChange(updatedRows);
    
    setShowAddDialog(false);
    setNewItem({ description: '', qty: 1, unit_price: 0, line_total: 0 });
  };

  const handleExportCSV = () => {
    const csvContent = [
      'Description,Quantity,Unit Price,Line Total,Category',
      ...rows.map(row => [
        `"${row.description}"`,
        row.qty,
        row.unit_price,
        row.line_total.toFixed(2),
        row.category || ''
      ].join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `extracted_data_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const totalAmount = rows.reduce((sum, row) => sum + row.line_total, 0);
  const totalItems = rows.length;

  return (
    <div className="w-full">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-lg border border-blue-200">
        <div>
          <h3 className="text-xl font-bold text-gray-900 mb-3">📊 Extracted Data</h3>
          <div className="flex flex-wrap gap-3">
            <span className="px-3 py-2 bg-blue-500 text-white text-sm font-semibold rounded-full shadow-sm">
              📦 {totalItems} items
            </span>
            <span className="px-3 py-2 bg-green-500 text-white text-sm font-semibold rounded-full shadow-sm">
              💰 Total: ${totalAmount.toFixed(2)}
            </span>
            {isModified && (
              <span className="px-3 py-2 bg-orange-500 text-white text-sm font-semibold rounded-full shadow-sm animate-pulse">
                ✏️ Modified
              </span>
            )}
          </div>
        </div>
        
        <div className="flex items-center gap-3 mt-4 sm:mt-0">
          <button
            onClick={handleExportCSV}
            className="p-3 text-gray-700 hover:text-blue-600 hover:bg-blue-50 transition-colors rounded-full border border-gray-300 shadow-sm"
            title="Export to CSV"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </button>
          
          {!isReadOnly && (
            <>
              <button
                onClick={() => setShowAddDialog(true)}
                className="btn btn-outline text-sm flex items-center font-semibold shadow-sm"
              >
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                </svg>
                Add Item
              </button>
              
              {isModified && (
                <button
                  onClick={() => {
                    onSave();
                    setIsModified(false);
                  }}
                  disabled={isLoading}
                  className="btn btn-primary text-sm flex items-center font-semibold shadow-lg"
                >
                  {isLoading ? (
                    <>
                      <div className="spinner w-4 h-4 mr-2"></div>
                      Saving...
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
                      </svg>
                      Save Changes
                    </>
                  )}
                </button>
              )}
            </>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="card overflow-hidden shadow-lg">
        <div className="overflow-x-auto">
          <table className="w-full bg-white">
            <thead>
              <tr className="bg-gradient-to-r from-gray-50 to-gray-100 border-b-2 border-gray-200">
                <th className="text-left py-4 px-6 font-bold text-gray-900 text-sm uppercase tracking-wide">Description</th>
                <th className="text-right py-4 px-6 font-bold text-gray-900 text-sm uppercase tracking-wide">Qty</th>
                <th className="text-right py-4 px-6 font-bold text-gray-900 text-sm uppercase tracking-wide">Unit Price</th>
                <th className="text-right py-4 px-6 font-bold text-gray-900 text-sm uppercase tracking-wide">Line Total</th>
                <th className="text-left py-4 px-6 font-bold text-gray-900 text-sm uppercase tracking-wide">Category</th>
                {!isReadOnly && (
                  <th className="text-center py-4 px-6 font-bold text-gray-900 text-sm uppercase tracking-wide">Actions</th>
                )}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {rows.map((row, index) => (
                <tr key={index} className="hover:bg-blue-50 transition-colors duration-150 border-b border-gray-100">
                  <td className="py-4 px-6">
                    {editingRow === index ? (
                      <input
                        type="text"
                        value={row.description}
                        onChange={(e) => handleFieldChange(index, 'description', e.target.value)}
                        className="input-field w-full font-medium text-gray-900"
                      />
                    ) : (
                      <div className="font-semibold text-gray-900 text-sm leading-relaxed">{row.description}</div>
                    )}
                  </td>
                  <td className="py-4 px-6 text-right">
                    {editingRow === index ? (
                      <input
                        type="number"
                        value={row.qty}
                        onChange={(e) => handleFieldChange(index, 'qty', Number(e.target.value))}
                        className="input-field w-20 text-right font-medium text-gray-900"
                        min="0"
                        step="0.01"
                      />
                    ) : (
                      <span className="font-semibold text-gray-900 text-sm">{row.qty}</span>
                    )}
                  </td>
                  <td className="py-4 px-6 text-right">
                    {editingRow === index ? (
                      <input
                        type="number"
                        value={row.unit_price}
                        onChange={(e) => handleFieldChange(index, 'unit_price', Number(e.target.value))}
                        className="input-field w-24 text-right font-medium text-gray-900"
                        min="0"
                        step="0.01"
                      />
                    ) : (
                      <span className="font-semibold text-gray-900 text-sm">${row.unit_price.toFixed(2)}</span>
                    )}
                  </td>
                  <td className="py-4 px-6 text-right">
                    <span className="font-bold text-green-700 text-sm bg-green-50 px-2 py-1 rounded">
                      ${row.line_total.toFixed(2)}
                    </span>
                  </td>
                  <td className="py-4 px-6">
                    {row.category && (
                      <span className="px-3 py-1 bg-blue-100 text-blue-800 text-xs font-semibold rounded-full">
                        {row.category}
                      </span>
                    )}
                  </td>
                  {!isReadOnly && (
                    <td className="py-4 px-6 text-center">
                      {editingRow === index ? (
                        <div className="flex items-center justify-center gap-3">
                          <button
                            onClick={() => handleSave(index)}
                            className="text-green-600 hover:text-green-700 hover:bg-green-50 p-2 rounded-full transition-colors"
                            title="Save"
                          >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                          </button>
                          <button
                            onClick={handleCancel}
                            className="text-red-600 hover:text-red-700 hover:bg-red-50 p-2 rounded-full transition-colors"
                            title="Cancel"
                          >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </button>
                        </div>
                      ) : (
                        <div className="flex items-center justify-center gap-3">
                          <button
                            onClick={() => handleEdit(index)}
                            className="text-blue-600 hover:text-blue-700 hover:bg-blue-50 p-2 rounded-full transition-colors"
                            title="Edit"
                          >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                            </svg>
                          </button>
                          <button
                            onClick={() => handleDelete(index)}
                            className="text-red-600 hover:text-red-700 hover:bg-red-50 p-2 rounded-full transition-colors"
                            title="Delete"
                          >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                          </button>
                        </div>
                      )}
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Summary */}
      <div className="bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-lg p-6 mt-6 shadow-sm">
        <div className="flex items-center">
          <svg className="w-6 h-6 mr-3 text-green-600" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
          <div className="flex-1">
            <div className="text-lg font-bold text-gray-900 mb-1">
              📊 Summary: {totalItems} items totaling <span className="text-green-700">${totalAmount.toFixed(2)}</span>
            </div>
            {!isReadOnly && (
              <p className="text-sm text-gray-700 font-medium">
                💡 Click the <span className="text-blue-600 font-semibold">edit button</span> to modify items or use the <span className="text-blue-600 font-semibold">Add button</span> to include new ones.
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Add Item Dialog */}
      {showAddDialog && (
        <div className="modal-overlay">
          <div className="modal-content border-2 border-blue-200 shadow-2xl">
            <div className="flex items-center mb-6">
              <div className="bg-blue-100 p-3 rounded-full mr-4">
                <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                </svg>
              </div>
              <h3 className="text-xl font-bold text-gray-900">Add New Item</h3>
            </div>
            <div className="space-y-5">
              <div>
                <label className="block text-sm font-bold text-gray-900 mb-2">
                  📝 Description *
                </label>
                <input
                  type="text"
                  value={newItem.description}
                  onChange={(e) => setNewItem(prev => ({ ...prev, description: e.target.value }))}
                  className="input-field font-medium text-gray-900"
                  placeholder="Enter item description"
                  required
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-bold text-gray-900 mb-2">
                    📦 Quantity
                  </label>
                  <input
                    type="number"
                    value={newItem.qty}
                    onChange={(e) => setNewItem(prev => ({ ...prev, qty: Number(e.target.value) }))}
                    className="input-field font-medium text-gray-900"
                    min="0.01"
                    step="0.01"
                  />
                </div>
                <div>
                  <label className="block text-sm font-bold text-gray-900 mb-2">
                    💰 Unit Price
                  </label>
                  <input
                    type="number"
                    value={newItem.unit_price}
                    onChange={(e) => setNewItem(prev => ({ ...prev, unit_price: Number(e.target.value) }))}
                    className="input-field font-medium text-gray-900"
                    min="0"
                    step="0.01"
                  />
                </div>
              </div>
              <div className="p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border border-green-200">
                <p className="text-sm font-bold text-gray-900">
                  💵 Line Total: <span className="text-green-700 text-lg">${((newItem.qty || 0) * (newItem.unit_price || 0)).toFixed(2)}</span>
                </p>
              </div>
            </div>
            <div className="flex justify-end gap-3 mt-8">
              <button
                onClick={() => setShowAddDialog(false)}
                className="btn btn-outline font-semibold"
              >
                Cancel
              </button>
              <button
                onClick={handleAddItem}
                disabled={!newItem.description?.trim()}
                className="btn btn-primary font-semibold shadow-lg"
              >
                ✅ Add Item
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EditableTable;