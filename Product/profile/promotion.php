<?php
//------------------ Promotion State --------------------------------------------------------------
echo "<br>";
echo "<h2>Promotion</h2>";
echo "<p><strong>Years Since Last Promotion:</strong> " . htmlspecialchars($row["YearsSinceLastPromotion"]) . "</p>";

// Selected features to pass
$features = ["Salary", "BusinessTravel", "MaritalStatus", "StockOptionLevel", "OverTime", "Age", "YearsSinceLastPromotion"];

$input_data = [];
foreach ($features as $feature) {
    $input_data[$feature] = $row[$feature];
} 

$input_data["YearsSinceLastPromotion"]=0;

include 'Attrition_Pred.php';

if ($prediction_result == "Yes") {
    echo "<div class='action-group'>";
    echo "<div class='action-description'>";
    echo "<p>Giving a promotion now (without salary increment or other changes) will set attrition status <strong>Yes</strong>.</p>";
    echo "</div>";

    echo "<div class='button-result-group'>";
    echo '<form method="post" class="inline-form">';
    echo '<input type="hidden" name="check_salary" value="salary_check_2">';
    echo '<button type="submit">Check Suggested Salary After Promotion</button>';
    echo '</form>';

    if (isset($_POST['check_salary']) && $_POST['check_salary'] === 'salary_check_2') {
        $features = ["Salary", "BusinessTravel", "MaritalStatus", "StockOptionLevel", "OverTime", "Age", "YearsSinceLastPromotion"];
        $input_data = [];
        foreach ($features as $feature) {
            $input_data[$feature] = $row[$feature];
        }
        $input_data['YearsSinceLastPromotion']=0;
        include 'Attrition_Salary_incr.php';
        echo "<p><strong>Suggested Salary Range:</strong> " . htmlspecialchars($prediction_range) . "</p>";
    }
    echo "</div>"; // Close button-result-group

    echo "<div class='button-result-group'>";
    echo '<form method="post" class="inline-form">';
    echo '<input type="hidden" name="check_StockOption" value="check_StockOption">';
    echo '<button type="submit">Check Suggested Stock Option After Promotion</button>';
    echo '</form>';
    if ($_POST['check_StockOption'] === 'check_StockOption') {
        $features = ["Salary", "BusinessTravel", "MaritalStatus", "StockOptionLevel", "OverTime", "Age", "YearsSinceLastPromotion"];
        $input_data = [];
        foreach ($features as $feature) {
            $input_data[$feature] = $row[$feature];
        }
        $input_data['YearsSinceLastPromotion']=0;
        include "StockOption.php";
    }
    echo "</div>"; // Close button-result-group

    echo "<div class='button-result-group'>";
    echo '<form method="post" class="inline-form">';
    echo '<input type="hidden" name="check_OverTime" value="check_OverTime">';
    echo '<button type="submit">Check Suggested Over Time level After Promotion</button>';
    echo '</form>';
    if ($_POST['check_OverTime'] === 'check_OverTime') {
        $features = ["Salary", "BusinessTravel", "MaritalStatus", "StockOptionLevel", "OverTime", "Age", "YearsSinceLastPromotion"];
        $input_data = [];
        foreach ($features as $feature) {
            $input_data[$feature] = $row[$feature];
        }
        $input_data['YearsSinceLastPromotion']=0;
        include "OverTime.php";
    }
    echo "</div>"; // Close button-result-group

    echo "<div class='button-result-group'>";
    echo '<form method="post" class="inline-form">';
    echo '<input type="hidden" name="check_BusinessTravel" value="check_BusinessTravel">';
    echo '<button type="submit">Check Suggested Business Travel level After Promotion</button>';
    echo '</form>';
    if ($_POST['check_BusinessTravel'] === 'check_BusinessTravel') {
        $features = ["Salary", "BusinessTravel", "MaritalStatus", "StockOptionLevel", "OverTime", "Age", "YearsSinceLastPromotion"];
        $input_data = [];
        foreach ($features as $feature) {
            $input_data[$feature] = $row[$feature];
        }
        $input_data['YearsSinceLastPromotion']=0;
        include "BusinessTravel.php";
    }
    echo "</div>"; // Close button-result-group

    echo "</div>"; // Close action-group

} else {
    echo "<p>Giving a promotion now is <strong>unlikely to cause attrition</strong>.</p>";
}
echo "</div>"; // Close action-section

