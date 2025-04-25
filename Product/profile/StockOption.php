<?php

// Possible stock option levels to test
$stock_option_levels = [0, 1, 2, 3];
$suggested_levels = [];
foreach ($stock_option_levels as $level) {
    $input_data['StockOptionLevel'] = $level;
    #echo "<pre>" . print_r($input_data, true) . "</pre>";

    include 'Attrition_Pred.php'; 

    if ($prediction_result == "No") {
        $suggested_levels[] = $level;
    }
}

if (!empty($suggested_levels) && count($suggested_levels) != 4) {
    echo "<p>To keep attrition 'No', consider changing Stock Option Level to one of the following:</p>";
    echo "<p><strong>" . htmlspecialchars($level) . "</strong></p>";
} else {
    echo "<p>Changing stock option level alone won't reduce attrition risk. Consider other features.</p>";
}
    
?>