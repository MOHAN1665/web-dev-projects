const mongoose = require('mongoose');

const incomeSchema = new mongoose.Schema({
  // Define your income schema fields here
  incomeName: {
    type: String,
    required: true,
  },
  incomeAmount: {
    type: Number,
    required: true,
  },
  incomeDate: {
    type: Date,
    default: Date.now,
  },
type: {
    type: String,
    default: 'income' // This can also be set based on your application logic
  },
});

const Income = mongoose.model('Income', incomeSchema);

module.exports = Income;
