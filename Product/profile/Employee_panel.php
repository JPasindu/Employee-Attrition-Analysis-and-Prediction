<?php 
echo "<h2>Employee summery</h2>";

//------------------ Over Time --------------------------------------------------------------
echo "<h3 style=\"margin-top: 20px; padding: 10px; border-top: 1px solid #ccc;\">Overtime</h3>";

$features = ["Salary", "BusinessTravel", "MaritalStatus", "StockOptionLevel", "OverTime", "Age", "YearsSinceLastPromotion"];
$input_data = [];
foreach ($features as $feature) {
    $input_data[$feature] = $row[$feature];
}

if ($input_data["OverTime"] == "Yes") {
    echo "<p>The employee is currently working overtime.</p>";
    $input_data["OverTime"] = "No";
    include 'Attrition_Pred.php';

    if ($row["Attrition"] == "Yes") {
        if ($prediction_result == "Yes") {
            echo "<p>Stopping overtime will <strong>not significantly reduce</strong> attrition risk.</p>";
        } else {
            echo "<p><span style='color:green;'>Stopping overtime will <strong>help reduce</strong> attrition risk.</span></p>";
        }
    } else {
        echo "<p><span style='color:green;'>The employee is currently handling overtime well.</span></p>";
    }
} else {
    echo "<p>The employee is currently <strong>not</strong> doing overtime.</p>";
    $input_data["OverTime"] = "Yes";
    include 'Attrition_Pred.php';

    if ($prediction_result == "Yes") {
        echo "<p><span style='color:red;'>If the employee starts doing overtime, attrition risk may <strong>increase</strong>.</span></p>";
    } else {
        echo "<p><span style='color:green;'>The employee can likely handle overtime without increasing attrition risk</span>.</p>";
    }
}

//------------------ Salary State --------------------------------------------------------------
echo "<h3 style=\"margin-top: 20px; padding: 10px; border-top: 1px solid #ccc;\">Salary</h3>";
echo "<p><strong>Current Salary:</strong> " . htmlspecialchars($row["Salary"]) . "</p>";

if ($row["Attrition"] == "Yes") {
    echo '<form method="post">';
    echo '<input type="hidden" name="check_salary" value="salary_check_1">';
    echo '<button type="submit">Check Suggested Salary to Reduce Attrition Risk</button>';
    echo '</form>';

    if (isset($_POST['check_salary']) && $_POST['check_salary'] === 'salary_check_1') {
        $features = ["Salary", "BusinessTravel", "MaritalStatus", "StockOptionLevel", "OverTime", "Age", "YearsSinceLastPromotion"];
        $input_data = [];
        foreach ($features as $feature) {
            $input_data[$feature] = $row[$feature];
        }

        include 'Attrition_Salary_incr.php';
        echo "<p><strong>Suggested Salary Range:</strong> " . htmlspecialchars($prediction_range) . "</p>";
    }
} else {
    echo "<p><strong>Reducing salary may increase the risk of attrition.</strong></p>";
    echo "<p><strong>Go to 'Change features to check attrition' section and test.</strong></p>";

    /*
    echo '<form method="post">';
    echo '<input type="hidden" name="salary_check_dec" value="salary_check_dec">';
    echo '<button type="submit">Check Salary reduction without risk of attrition</button>';
    echo '</form>';

    if ($_POST['salary_check_dec'] === 'salary_check_dec') {
        $features = ["Salary", "BusinessTravel", "MaritalStatus", "StockOptionLevel", "OverTime", "Age", "YearsSinceLastPromotion"];
        $input_data = [];
        foreach ($features as $feature) {
            $input_data[$feature] = $row[$feature];
        }

        include 'Attrition_Salary_dec.php';
        echo "<p><strong>Suggested Salary Range:</strong> " . htmlspecialchars($prediction_range) . "</p>";
    }
    */
}

//------------------ Stock Option --------------------------------------------------------------
echo "<h3 style=\"margin-top: 20px; padding: 10px; border-top: 1px solid #ccc;\">Stock Option</h3>";
echo "<p><strong>Current Stock Option Level:</strong> " . htmlspecialchars($row["StockOptionLevel"]) . "</p>";

$features = ["Salary", "BusinessTravel", "MaritalStatus", "StockOptionLevel", "OverTime", "Age", "YearsSinceLastPromotion"];
$input_data = [];
foreach ($features as $feature) {
    $input_data[$feature] = $row[$feature];
}
if ($row['Attrition'] == "Yes") {
    include "StockOption.php";
} else {
    echo "<p><strong>Changing stock option level may affect attrition.</strong></p>";
    echo "<p><strong>Go to 'Change features to check attrition' section and test.</strong></p>";
}

//------------------ BusinessTravel Option --------------------------------------------------------------
echo "<h3 style=\"margin-top: 20px; padding: 10px; border-top: 1px solid #ccc;\">Business Travel</h3>";
echo "<p><strong>Current Business Travel Level:</strong> " . htmlspecialchars($row["BusinessTravel"]) . "</p>";

$features = ["Salary", "BusinessTravel", "MaritalStatus", "StockOptionLevel", "OverTime", "Age", "YearsSinceLastPromotion"];
$input_data = [];
foreach ($features as $feature) {
    $input_data[$feature] = $row[$feature];
}

// Possible stock option levels to test
if ($row['Attrition'] == "Yes") {
    include "BusinessTravel.php";
} else {
    echo "<p><strong>Changing BusinessTravel level may affect attrition.</strong></p>";
    echo "<p><strong>Go to 'Change features to check attrition' section and test.</strong></p>";
}


?>
