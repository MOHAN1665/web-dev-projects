const expenseForm = document.getElementById("expense-form");
const expenseList = document.getElementById("expense-list");
const totalAmountElement = document.getElementById("total-amount");
let expenses = [];

async function fetchAndRenderExpenses() {
  try {
    const response = await fetch('/expenses');
    if (response.ok) {
      expenses = await response.json();
      renderExpenses();
    } else {
      throw new Error('Failed to fetch expenses');
    }
  } catch (error) {
    console.error('Error fetching expenses:', error.message);
  }
}

function renderExpenses() {
  expenseList.innerHTML = "";
  let totalAmount = 0;

  expenses.forEach((expense, i) => {
    const expenseRow = document.createElement("tr");

	const date = new Date(expense.date);
  	const formattedDate = date.toLocaleDateString('en-IN');

    expenseRow.innerHTML = `
      <td>${expense.name}</td>
      <td>$${expense.amount}</td>
      <td>${formattedDate}</td>
      <td class="delete-btn" data-id="${i}">Delete</td>
    `;
    expenseList.appendChild(expenseRow);

    totalAmount += expense.amount;
  });

  totalAmountElement.textContent = totalAmount.toFixed(2);
}

async function addExpense(event) {
  event.preventDefault();

  const expenseNameInput = document.getElementById("expense-name");
  const expenseAmountInput = document.getElementById("expense-amount");
  const expenseDateInput = document.getElementById("expense-date");

  const name = expenseNameInput.value;
  const amount = parseFloat(expenseAmountInput.value);
  const date = expenseDateInput.value;

  if (!name || isNaN(amount) || !date) {
    alert("Please enter valid expense details.");
    return;
  }

  try {
    const response = await fetch('/expenses', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name, amount, date }),
    });

    if (response.ok) {
      const newExpense = await response.json();
      expenses.push(newExpense);
      renderExpenses();

      // Reset the form after adding an expense
      expenseNameInput.value = '';
      expenseAmountInput.value = '';
      expenseDateInput.value = '';
    } else {
      throw new Error('Failed to add expense');
    }
  } catch (error) {
    console.error('Error adding expense:', error.message);
  }
}

async function deleteExpense(event) {
  if (event.target.classList.contains("delete-btn")) {
    const expenseIndex = parseInt(event.target.getAttribute("data-id"));
    const deletedExpense = expenses[expenseIndex];

    try {
      if (!deletedExpense || !deletedExpense._id) {
        throw new Error('Invalid expense or expense ID');
      }

      const response = await fetch(`/expenses/${deletedExpense._id}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        await fetchAndRenderExpenses(); // Fetch and re-render expenses after successful deletion
      } else {
        throw new Error('Failed to delete expense');
      }
    } catch (error) {
      console.error('Error deleting expense:', error.message);
    }
  }
}


// Event listener for deleting expenses
expenseList.addEventListener("click", deleteExpense);


// Sort expenses by date
function sortExpensesByDate() {
  expenses.sort((a, b) => new Date(a.date) - new Date(b.date));
  renderExpenses();
}

// Sort expenses by amount
function sortExpensesByAmount() {
  expenses.sort((a, b) => a.amount - b.amount);
  renderExpenses();
}

expenseForm.addEventListener("submit", addExpense);
expenseList.addEventListener("click", deleteExpense);

// Event listeners for sorting buttons
document.getElementById("sort-date-btn").addEventListener("click", sortExpensesByDate);
document.getElementById("sort-amount-btn").addEventListener("click", sortExpensesByAmount);

document.addEventListener("DOMContentLoaded", () => {
  fetchAndRenderExpenses();
});
