import React, { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Alert,
  Chip,
  IconButton,
  Tooltip,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import {
  DataGrid,
  GridColDef,
  GridRowId,
  GridRowModel,
  GridRowsProp,
  GridActionsCellItem,
  GridEventListener,
  GridRowEditStopReasons,
} from '@mui/x-data-grid';
import {
  Edit as EditIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  Download as DownloadIcon
} from '@mui/icons-material';
import { EditableTableProps, Item } from '../types/schema';

interface EditableTableState {
  rows: GridRowsProp;
  editingRow: GridRowId | null;
  isModified: boolean;
}

const EditableTable: React.FC<EditableTableProps> = ({
  data,
  onDataChange,
  onSave,
  isReadOnly = false,
  isLoading = false
}) => {
  const [tableState, setTableState] = useState<EditableTableState>({
    rows: [],
    editingRow: null,
    isModified: false
  });
  const [addRowDialog, setAddRowDialog] = useState(false);
  const [newRowData, setNewRowData] = useState<Partial<Item>>({
    description: '',
    qty: 1,
    unit_price: 0,
    line_total: 0
  });

  // Convert data to DataGrid format
  useEffect(() => {
    const rows = data.map((item, index) => ({
      id: index,
      ...item
    }));
    setTableState(prev => ({ ...prev, rows }));
  }, [data]);

  // Column definitions
  const columns: GridColDef[] = [
    {
      field: 'description',
      headerName: 'Description',
      flex: 2,
      minWidth: 200,
      editable: !isReadOnly,
      renderCell: (params) => (
        <Box sx={{ 
          overflow: 'hidden', 
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          width: '100%'
        }}>
          {params.value}
        </Box>
      )
    },
    {
      field: 'qty',
      headerName: 'Quantity',
      type: 'number',
      width: 100,
      editable: !isReadOnly,
      align: 'right',
      headerAlign: 'right',
      valueFormatter: (params) => {
        if (params.value == null) return '';
        return Number(params.value).toLocaleString();
      }
    },
    {
      field: 'unit_price',
      headerName: 'Unit Price',
      type: 'number',
      width: 120,
      editable: !isReadOnly,
      align: 'right',
      headerAlign: 'right',
      valueFormatter: (params) => {
        if (params.value == null) return '';
        return `$${Number(params.value).toFixed(2)}`;
      }
    },
    {
      field: 'line_total',
      headerName: 'Line Total',
      type: 'number',
      width: 120,
      editable: false,
      align: 'right',
      headerAlign: 'right',
      valueGetter: (params) => {
        const qty = params.row.qty || 0;
        const unitPrice = params.row.unit_price || 0;
        return (qty * unitPrice);
      },
      valueFormatter: (params) => {
        if (params.value == null) return '';
        return `$${Number(params.value).toFixed(2)}`;
      },
      cellClassName: 'font-semibold bg-gray-50'
    },
    {
      field: 'category',
      headerName: 'Category',
      width: 130,
      editable: false,
      renderCell: (params) => {
        if (!params.value) return null;
        return (
          <Chip 
            label={params.value} 
            size="small" 
            color="primary" 
            variant="outlined"
          />
        );
      }
    }
  ];

  // Add actions column if not readonly
  if (!isReadOnly) {
    columns.push({
      field: 'actions',
      type: 'actions',
      headerName: 'Actions',
      width: 100,
      getActions: ({ id }) => [
        <GridActionsCellItem
          key={id}
          icon={<DeleteIcon />}
          label="Delete"
          onClick={() => handleDeleteRow(id)}
          color="inherit"
        />,
      ],
    });
  }

  const handleDeleteRow = (id: GridRowId) => {
    const updatedRows = tableState.rows.filter(row => row.id !== id);
    setTableState(prev => ({ 
      ...prev, 
      rows: updatedRows, 
      isModified: true 
    }));
    
    const updatedData = updatedRows.map(row => ({
      description: row.description,
      qty: row.qty,
      unit_price: row.unit_price,
      line_total: row.qty * row.unit_price,
      category: row.category
    }));
    onDataChange(updatedData);
  };

  const handleRowEditStop: GridEventListener<'rowEditStop'> = (params, event) => {
    if (params.reason === GridRowEditStopReasons.rowFocusOut) {
      event.defaultMuiPrevented = true;
    }
  };

  const processRowUpdate = useCallback((newRow: GridRowModel) => {
    // Calculate line total
    newRow.line_total = (newRow.qty || 0) * (newRow.unit_price || 0);
    
    const updatedRows = tableState.rows.map(row => 
      row.id === newRow.id ? newRow : row
    );
    
    setTableState(prev => ({ 
      ...prev, 
      rows: updatedRows, 
      isModified: true 
    }));
    
    const updatedData = updatedRows.map(row => ({
      description: row.description || '',
      qty: row.qty || 0,
      unit_price: row.unit_price || 0,
      line_total: (row.qty || 0) * (row.unit_price || 0),
      category: row.category
    }));
    onDataChange(updatedData);
    
    return newRow;
  }, [tableState.rows, onDataChange]);

  const handleAddRow = () => {
    if (!newRowData.description?.trim()) return;
    
    const lineTotal = (newRowData.qty || 0) * (newRowData.unit_price || 0);
    const newRow = {
      id: Math.max(...tableState.rows.map(r => Number(r.id)), -1) + 1,
      description: newRowData.description,
      qty: newRowData.qty || 1,
      unit_price: newRowData.unit_price || 0,
      line_total: lineTotal,
      category: newRowData.category
    };
    
    const updatedRows = [...tableState.rows, newRow];
    setTableState(prev => ({ 
      ...prev, 
      rows: updatedRows, 
      isModified: true 
    }));
    
    const updatedData = updatedRows.map(row => ({
      description: row.description,
      qty: row.qty,
      unit_price: row.unit_price,
      line_total: row.qty * row.unit_price,
      category: row.category
    }));
    onDataChange(updatedData);
    
    setAddRowDialog(false);
    setNewRowData({ description: '', qty: 1, unit_price: 0, line_total: 0 });
  };

  const handleExportCSV = () => {
    const csvContent = [
      'Description,Quantity,Unit Price,Line Total,Category',
      ...tableState.rows.map(row => [
        `"${row.description}"`,
        row.qty,
        row.unit_price,
        (row.qty * row.unit_price).toFixed(2),
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

  // Calculate totals
  const totalAmount = tableState.rows.reduce(
    (sum, row) => sum + ((row.qty || 0) * (row.unit_price || 0)), 
    0
  );
  const totalItems = tableState.rows.length;

  return (
    <Box className="w-full">
      {/* Header */}
      <Box 
        display="flex" 
        justifyContent="space-between" 
        alignItems="center" 
        mb={2}
      >
        <Box>
          <Typography variant="h6" gutterBottom>
            Extracted Data
          </Typography>
          <Box display="flex" gap={2}>
            <Chip 
              label={`${totalItems} items`} 
              color="primary" 
              size="small"
            />
            <Chip 
              label={`Total: $${totalAmount.toFixed(2)}`} 
              color="success" 
              size="small"
            />
            {tableState.isModified && (
              <Chip 
                label="Modified" 
                color="warning" 
                size="small"
              />
            )}
          </Box>
        </Box>
        
        <Box display="flex" gap={1}>
          <Tooltip title="Export to CSV">
            <IconButton onClick={handleExportCSV} color="primary">
              <DownloadIcon />
            </IconButton>
          </Tooltip>
          
          {!isReadOnly && (
            <>
              <Button
                startIcon={<AddIcon />}
                onClick={() => setAddRowDialog(true)}
                variant="outlined"
                size="small"
              >
                Add Item
              </Button>
              
              {tableState.isModified && (
                <Button
                  startIcon={<SaveIcon />}
                  onClick={() => {
                    onSave();
                    setTableState(prev => ({ ...prev, isModified: false }));
                  }}
                  variant="contained"
                  color="primary"
                  size="small"
                  disabled={isLoading}
                >
                  Save Changes
                </Button>
              )}
            </>
          )}
        </Box>
      </Box>

      {/* Data Grid */}
      <Paper elevation={1}>
        <DataGrid
          rows={tableState.rows}
          columns={columns}
          editMode="row"
          onRowEditStop={handleRowEditStop}
          processRowUpdate={processRowUpdate}
          pageSizeOptions={[5, 10, 25, 50]}
          initialState={{
            pagination: {
              paginationModel: { pageSize: 10 }
            }
          }}
          checkboxSelection={false}
          disableRowSelectionOnClick
          loading={isLoading}
          autoHeight
          sx={{
            '& .MuiDataGrid-cell': {
              borderRight: '1px solid #f0f0f0'
            },
            '& .MuiDataGrid-columnHeaders': {
              backgroundColor: '#f8fafc',
              fontWeight: 600
            },
            '& .font-semibold': {
              fontWeight: 600
            },
            '& .bg-gray-50': {
              backgroundColor: '#f9fafb'
            }
          }}
        />
      </Paper>

      {/* Summary */}
      <Box mt={2}>
        <Alert severity="info">
          <Typography variant="body2">
            <strong>Summary:</strong> {totalItems} items totaling ${totalAmount.toFixed(2)}.
            {!isReadOnly && ' You can edit cells by double-clicking or add new items using the Add button.'}
          </Typography>
        </Alert>
      </Box>

      {/* Add Row Dialog */}
      <Dialog 
        open={addRowDialog} 
        onClose={() => setAddRowDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Add New Item</DialogTitle>
        <DialogContent>
          <Box display="flex" flexDirection="column" gap={2} mt={1}>
            <TextField
              label="Description"
              value={newRowData.description}
              onChange={(e) => setNewRowData(prev => ({ ...prev, description: e.target.value }))}
              fullWidth
              required
            />
            <TextField
              label="Quantity"
              type="number"
              value={newRowData.qty}
              onChange={(e) => setNewRowData(prev => ({ ...prev, qty: Number(e.target.value) }))}
              inputProps={{ min: 0.01, step: 0.01 }}
            />
            <TextField
              label="Unit Price"
              type="number"
              value={newRowData.unit_price}
              onChange={(e) => setNewRowData(prev => ({ ...prev, unit_price: Number(e.target.value) }))}
              inputProps={{ min: 0, step: 0.01 }}
            />
            <Box p={2} bgcolor="#f8fafc" borderRadius={1}>
              <Typography variant="body2" color="textSecondary">
                Line Total: ${((newRowData.qty || 0) * (newRowData.unit_price || 0)).toFixed(2)}
              </Typography>
            </Box>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddRowDialog(false)}>
            Cancel
          </Button>
          <Button 
            onClick={handleAddRow}
            variant="contained"
            disabled={!newRowData.description?.trim()}
          >
            Add Item
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default EditableTable;