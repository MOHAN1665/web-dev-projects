const express = require('express');
const router = express.Router();
const Expense = require('../models/Expense');

// GET all expenses
router.get('/', async (req, res) => {
  try {
    const expenses = await Expense.find();
    res.json(expenses);
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

// POST a new expense
router.post('/', async (req, res) => {
  const { name, amount, date } = req.body;

  if (!name || !amount || !date) {
    return res.status(400).json({ message: "Please provide name, amount, and date for the expense." });
  }

  try {
    const newExpense = await Expense.create({
      name,
      amount,
      date,
    });

    res.status(201).json(newExpense);
  } catch (err) {
    console.error('Error adding expense:', err.message);
    res.status(500).json({ message: 'Failed to add expense.' });
  }
});

// DELETE an expense
router.delete('/:id', async (req, res) => {
  try {
    const deletedExpense = await Expense.findByIdAndDelete(req.params.id);
    if (!deletedExpense) {
      return res.status(404).json({ message: 'Expense not found' });
    }
    res.json({ message: 'Expense deleted' });
  } catch (err) {
    console.error('Error deleting expense:', err.message);
    res.status(500).json({ message: 'Failed to delete expense' });
  }
});

module.exports = router;
