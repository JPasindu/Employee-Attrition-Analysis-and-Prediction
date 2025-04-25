<?php
// Possible OverTime values to test
    $overtime_levels = ["Yes", "No"];
    $suggested_levels = [];

    foreach ($overtime_levels as $level) {
        $input_data['OverTime'] = $level;
        // Uncomment this for debugging:
        // echo "<pre>" . print_r($input_data, true) . "</pre>";

        include 'Attrition_Pred.php'; // Must return prediction in $prediction_result

        if ($prediction_result == "No") {
            $suggested_levels[] = $level;
        }
    }

    if (!empty($suggested_levels)) {
        echo "<p>To keep attrition 'No', consider this over time status:";
        echo "<strong>" . htmlspecialchars(implode(', ', $suggested_levels)) . "</strong></p>";
    } else {
        echo "<p>Changing OverTime status alone won't reduce attrition risk. Consider other features.</p>";
    }
?>