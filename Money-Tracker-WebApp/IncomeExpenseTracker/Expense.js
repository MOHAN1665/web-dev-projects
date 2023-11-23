const mongoose = require('mongoose');

const expenseSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
  },
  amount: {
    type: Number,
    required: true,
  },
  date: {
    type: Date,
    default: Date.now,
  },
type: {
    type: String,
    default: 'expenses' // This can also be set based on your application logic
  },
});

module.exports = mongoose.model('Expense', expenseSchema);
