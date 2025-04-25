
<?php
$BusinessTravelLevels = ['No Travel', 'Some Travel', 'Frequent Traveller'];

    $suggested_levels = [];

    foreach ($BusinessTravelLevels as $level) {
        $input_data['BusinessTravel'] = $level;
        #echo "<pre>" . print_r($input_data, true) . "</pre>";

        include 'Attrition_Pred.php'; 

        if ($prediction_result == "No") {
            $suggested_levels[] = $level;
        }
    }
    if (!empty($suggested_levels) && count($suggested_levels) != 3) {
        echo "<p>To keep attrition 'No', consider changing BusinessTravel Level to one of the following:</p>";
        echo "<p><strong>" . htmlspecialchars(implode(', ', $suggested_levels)) . "</strong></p>";
    } else {
        echo "<p>Changing BusinessTravel level alone won't reduce attrition risk. Consider other features.</p>";
    }
?>