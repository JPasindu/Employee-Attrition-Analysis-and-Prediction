<?php include('../navbar.html'); ?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Employee Records</title>
    <link rel="stylesheet" href="styles/style_view_dataset.css">
</head>
<body></body>

<?php

include '../connection.php';

// Enable error reporting
mysqli_report(MYSQLI_REPORT_ERROR | MYSQLI_REPORT_STRICT);

try {
    // Capture and trim filters
    $search_id = trim($_GET['search_id'] ?? '');
    $state = trim($_GET['state'] ?? '');
    $department = trim($_GET['department'] ?? '');
    $jobrole = trim($_GET['jobrole'] ?? '');
    $attrition = trim($_GET['attrition'] ?? '');

    // Build WHERE clause
    $conditions = [];
    $params = [];
    $types = '';

    if (!empty($search_id)) {
        $conditions[] = "EmployeeID LIKE ?";
        $params[] = '%' . $search_id . '%';
        $types .= 's';
    }
    if (!empty($state)) {
        $conditions[] = "State = ?";
        $params[] = $state;
        $types .= 's';
    }
    if (!empty($department)) {
        $conditions[] = "Department = ?";
        $params[] = $department;
        $types .= 's';
    }
    if (!empty($jobrole)) {
        $conditions[] = "JobRole = ?";
        $params[] = $jobrole;
        $types .= 's';
    }
    if (!empty($attrition)) {
        $conditions[] = "Attrition = ?";
        $params[] = $attrition;
        $types .= 's';
    }

    // Prepare SQL query
    $sql = "SELECT EmployeeID,FirstName,State,Department,JobRole,Age,Gender,Attrition FROM EmployeeData";
    if (!empty($conditions)) {
        $sql .= " WHERE " . implode(" AND ", $conditions);
    }

    // Prepare and execute statement
    $stmt = mysqli_prepare($conn, $sql);
    if (!empty($params)) {
        mysqli_stmt_bind_param($stmt, $types, ...$params);
    }

    // ... rest of your code ...
    mysqli_stmt_execute($stmt);
    $result = mysqli_stmt_get_result($stmt);

} catch (Exception $e) {
    die("Error: " . $e->getMessage());
}
?>

<h2>Employee Records</h2>

<form method="GET" action="">
    <!-- Add this search field at the top of your form -->
    <label>Search Employee ID:</label>
    <input type="text" name="search_id" value="<?php echo htmlspecialchars($_GET['search_id'] ?? ''); ?>" placeholder="Enter Employee ID">
    <!-- Dynamically populated State dropdown -->
    <label>State:</label>
    <select name="state">
        <option value="">-- All --</option>
        <?php
        $states = mysqli_query($conn, "SELECT DISTINCT State FROM EmployeeData ORDER BY State");
        while ($row = mysqli_fetch_assoc($states)) {
            $selected = $state === $row['State'] ? 'selected' : '';
            echo "<option value='{$row['State']}' $selected>{$row['State']}</option>";
        }
        ?>
    </select>

    <!-- Repeat similar dynamic dropdowns for Department, JobRole, and Attrition -->
    <!-- Department Dropdown -->
    <label>Department:</label>
    <select name="department">
        <option value="">-- All --</option>
        <?php
        $depts = mysqli_query($conn, "SELECT DISTINCT Department FROM EmployeeData ORDER BY Department");
        while ($row = mysqli_fetch_assoc($depts)) {
            $selected = $department === $row['Department'] ? 'selected' : '';
            echo "<option value='{$row['Department']}' $selected>{$row['Department']}</option>";
        }
        ?>
    </select>

    <!-- Job Role Dropdown -->
    <label>Job Role:</label>
    <select name="jobrole">
        <option value="">-- All --</option>
        <?php
        $roles = mysqli_query($conn, "SELECT DISTINCT JobRole FROM EmployeeData ORDER BY JobRole");
        while ($row = mysqli_fetch_assoc($roles)) {
            $selected = $jobrole === $row['JobRole'] ? 'selected' : '';
            echo "<option value='{$row['JobRole']}' $selected>{$row['JobRole']}</option>";
        }
        ?>
    </select>

    <!-- Attrition Dropdown -->
    <label>Attrition:</label>
    <select name="attrition">
        <option value="">-- All --</option>
        <?php
        $attritionOptions = ['Yes', 'No'];
        foreach ($attritionOptions as $option) {
            $selected = $attrition === $option ? 'selected' : '';
            echo "<option value='$option' $selected>$option</option>";
        }
        ?>
    </select>

</form>

<script>
    // Auto-submit form on dropdown change
    document.querySelectorAll('select').forEach(select => {
        select.addEventListener('change', () => {
            select.form.submit();
        });
    });

    // Add this to your existing script
    document.querySelector('input[name="search_id"]').addEventListener('input', function(e) {
        // Add a small delay to prevent too many requests while typing
        clearTimeout(this.timer);
        this.timer = setTimeout(() => {
            this.form.submit();
        }, 1500);
    });

    // Update the updateSummary function to include search_id
    function updateSummary() {
        const search_id = document.querySelector('input[name="search_id"]').value;
        const state = document.querySelector('select[name="state"]').value;
        const department = document.querySelector('select[name="department"]').value;
        const jobrole = document.querySelector('select[name="jobrole"]').value;
        const attrition = document.querySelector('select[name="attrition"]').value;

        const params = new URLSearchParams({ search_id, state, department, jobrole, attrition });

        fetch('summary.php?' + params.toString())
            .then(res => res.text())
            .then(html => {
                document.getElementById('summary').innerHTML = html;
            });
    }
</script>

<div style="display: flex; align-items: flex-start; gap: 20px;">
    <div style="flex: 1;">
        <!-- Results Table -->
        <?php
        if (mysqli_num_rows($result) > 0) {
            echo "<table border='1' cellpadding='10'>";
            // Header row
            echo "<tr>";
            while ($field = mysqli_fetch_field($result)) {
                echo "<th>{$field->name}</th>";
            }
            echo "</tr>";
            
            // Data rows
            while ($row = mysqli_fetch_assoc($result)) {
                $empId = $row['EmployeeID'];
                echo "<tr onclick=\"window.location.href='../profile/profile.php?id=$empId'\" style=\"cursor: pointer;\">";
                foreach ($row as $value) {
                    echo "<td>" . htmlspecialchars($value) . "</td>";
                }
                echo "</tr>";
            }
            
            
            echo "</table>";
        } else {
            echo "<p>No matching records found.</p>";
        }
        ?>
    </div>

    <div id="summary" style="width: 30%;">
        <!-- Summary content will be loaded here by JS -->
    </div>
</div>

<script>
function updateSummary() {
    const state = document.querySelector('select[name="state"]').value;
    const department = document.querySelector('select[name="department"]').value;
    const jobrole = document.querySelector('select[name="jobrole"]').value;
    const attrition = document.querySelector('select[name="attrition"]').value;

    const params = new URLSearchParams({ state, department, jobrole, attrition });

    fetch('summary.php?' + params.toString())
        .then(res => res.text())
        .then(html => {
            document.getElementById('summary').innerHTML = html;
        });
}

// Trigger summary update on filter change
['state', 'department', 'jobrole', 'attrition'].forEach(name => {
    document.querySelector(`select[name="${name}"]`).addEventListener('change', updateSummary);
});

// Initial load
updateSummary();
</script>


<?php
mysqli_close($conn);
?>
