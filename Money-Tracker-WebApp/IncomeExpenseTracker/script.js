document.addEventListener("DOMContentLoaded", () => {
  const expenseFields = document.getElementById("expense-fields");
  const incomeFields = document.getElementById("income-fields");
  const expenseOption = document.getElementById("expense-option");
  const incomeOption = document.getElementById("income-option");
  const totalAmountElement = document.getElementById("total-amount");

  function displayFields(option) {
    if (option === 'expenses') {
      expenseFields.style.display = "block";
      incomeFields.style.display = "none";
    } else if (option === 'income') {
      incomeFields.style.display = "block";
      expenseFields.style.display = "none";
    }
  }

  expenseOption.addEventListener("change", () => {
    if (expenseOption.checked) {
      displayFields('expenses');
      fetchEntries('expenses');
    }
  });

  incomeOption.addEventListener("change", () => {
    if (incomeOption.checked) {
      displayFields('income');
      fetchEntries('income');
    }
  });

  // Function to fetch entries and display fields initially
  async function init() {
    try {
      const response = await fetch('/expenses'); // Fetching expenses initially
      if (!response.ok) {
        throw new Error('Failed to fetch expenses');
      }
      entries = await response.json();
      displayEntries(entries);
      displayFields('expenses'); // Show expense fields initially
    } catch (error) {
      console.error('Error initializing:', error.message);
    }
  }

  init();

  async function fetchEntries(type) {
    try {
      const response = await fetch(`/${type}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch ${type}`);
      }
      entries = await response.json();
      displayEntries(entries);
    } catch (error) {
      console.error(`Error fetching ${type}:`, error.message);
    }
  }

  function displayEntries(entries) {
    const entryList = document.getElementById("entry-list");
    entryList.innerHTML = "";
    let totalAmount = 0;

    entries.forEach(entry => {
      let name, amount, date;

      if ('name' in entry && 'amount' in entry && 'date' in entry) {
        // This is for the directly inserted documents
        name = entry.name;
        amount = entry.amount;
        date = new Date(entry.date).toLocaleDateString();
      } else if ('incomeName' in entry && 'incomeAmount' in entry && 'incomeDate' in entry) {
        // This is for the frontend form-inserted documents
        name = entry.incomeName;
        amount = entry.incomeAmount;
        date = new Date(entry.incomeDate).toLocaleDateString();
      }

      const row = `
        <tr>
          <td>${name}</td>
          <td>${amount}</td>
          <td>${date}</td>
          <td>
            <button class="delete-button" data-id="${entry._id}" data-type="${entry.type}">Delete</button>
          </td>
        </tr>
      `;
      entryList.insertAdjacentHTML('beforeend', row);
      totalAmount += parseFloat(amount);
    });

    totalAmountElement.textContent = totalAmount.toFixed(2);
  }


  async function deleteEntry(id, type) {
  try {
    const response = await fetch(`/${type}/${id}`, {
      method: 'DELETE'
    });
    if (!response.ok) {
      throw new Error('Failed to delete entry');
    }
    fetchEntries(type);
  } catch (error) {
    console.error('Error deleting entry:', error.message);
  }
}

const entryForm = document.getElementById("entry-form");
entryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const type = expenseOption.checked ? 'expenses' : 'income';
  const nameInput = document.getElementById(`${type}-name`);
  const amountInput = document.getElementById(`${type}-amount`);
  const dateInput = document.getElementById(`${type}-date`);

  const name = nameInput.value;
  const amount = parseFloat(amountInput.value);
  const date = dateInput.value;

  let formData = {};

  // Check the type and set the form data accordingly
  if (type === 'expenses') {
    formData = { name, amount, date };
  } else if (type === 'income') {
    formData = { incomeName: name, incomeAmount: amount, incomeDate: date };
  }

  try {
    const response = await fetch(`/${type}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(formData)
    });
    if (!response.ok) {
      throw new Error('Failed to add entry');
    }
    fetchEntries(type);
    nameInput.value = "";
    amountInput.value = "";
    dateInput.value = "";
  } catch (error) {
    console.error('Error adding entry:', error.message);
  }
});
  // Event listener for the delete button
  document.addEventListener("click", async (event) => {
    if (event.target.classList.contains("delete-button")) {
      const id = event.target.dataset.id;
      const type = event.target.dataset.type;
      await deleteEntry(id, type);
    }
  });

 // Sort by date function
 function sortByDate(entries) {
  let dateField;
  if (expenseOption.checked) {
    dateField = 'date'; // Expense date field name
  } else {
    dateField = 'incomeDate'; // Income date field name
  }

  entries.sort((a, b) => new Date(a[dateField]) - new Date(b[dateField]));
  displayEntries(entries);
}

// Sort by amount function
function sortByAmount(entries) {
  let amountField;
  if (expenseOption.checked) {
    amountField = 'amount'; // Expense amount field name
  } else {
    amountField = 'incomeAmount'; // Income amount field name
  }

  entries.sort((a, b) => a[amountField] - b[amountField]);
  displayEntries(entries);
}


  // Event listeners for sorting buttons
  document.getElementById("sort-date-btn").addEventListener("click", () => {
    const type = expenseOption.checked ? 'expenses' : 'income';
    const dateField = type === 'expenses' ? 'date' : 'incomeDate';
    sortByDate(entries, dateField);
  });

  document.getElementById("sort-amount-btn").addEventListener("click", () => {
    const type = expenseOption.checked ? 'expenses' : 'income';
    const amountField = type === 'expenses' ? 'amount' : 'incomeAmount';
    sortByAmount(entries, amountField);
  });
// Event listeners for sorting buttons
document.getElementById("sort-date-btn").addEventListener("click", sortByDate);
document.getElementById("sort-amount-btn").addEventListener("click", sortByAmount);

  // Fetch and display entries initially based on the selected option
  if (expenseOption.checked) {
    fetchEntries('expenses');
  } else {
    fetchEntries('income');
  }
});
