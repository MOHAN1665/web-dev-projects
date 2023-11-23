const express = require('express');
const router = express.Router();
const Income = require('../models/Income');

// GET all incomes
router.get('/', async (req, res) => {
  try {
    const incomes = await Income.find();
    res.json(incomes);
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

// POST a new income
router.post('/', async (req, res) => {
  const { incomeName, incomeAmount, incomeDate } = req.body;

  if (!incomeName || !incomeAmount || !incomeDate) {
    return res.status(400).json({ message: "Please provide name, amount, and date for the income." });
  }

  try {
    const newIncome = await Income.create({
      incomeName,
      incomeAmount,
      incomeDate,
    });

    res.status(201).json(newIncome);
  } catch (err) {
    console.error('Error adding income:', err.message);
    res.status(500).json({ message: 'Failed to add income.' });
  }
});

// DELETE an income
router.delete('/:id', async (req, res) => {
  try {
    const deletedIncome = await Income.findByIdAndDelete(req.params.id);
    if (!deletedIncome) {
      return res.status(404).json({ message: 'Income not found' });
    }
    res.json({ message: 'Income deleted' });
  } catch (err) {
    console.error('Error deleting income:', err.message);
    res.status(500).json({ message: 'Failed to delete income' });
  }
});

module.exports = router;
