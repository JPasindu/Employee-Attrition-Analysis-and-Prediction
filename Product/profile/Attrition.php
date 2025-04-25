<?php

$changable_features = ["BusinessTravel", "StockOptionLevel", "OverTime", "Salary"];
$features = ["Salary", "BusinessTravel", "MaritalStatus", "StockOptionLevel", "OverTime", "Age", "YearsSinceLastPromotion"];
$prediction_result = '';
$predicted = false;

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['predict'])) {
    // Build feature input with posted values or fallback to DB values
    $input_data = [];
    foreach ($features as $feature) {
        if (in_array($feature, $changable_features)) {
            $input_data[$feature] = $_POST[$feature] ?? $row[$feature];
        } else {
            $input_data[$feature] = $row[$feature]; // static features from DB
        }
    }

    // Convert to JSON
    $json_data = json_encode($input_data);

    // Python script path
    // $command = "/Users/pasindumadusanka/myenv/bin/python3 predict_attrition.py";
    include "../python_connection.php";

    $command = $path."predict_attrition.py";

    $descriptorspec = [
        0 => ["pipe", "r"],  // stdin
        1 => ["pipe", "w"],  // stdout
        2 => ["pipe", "w"]   // stderr
    ];

    $process = proc_open($command, $descriptorspec, $pipes);

    if (is_resource($process)) {
        // Send JSON to stdin
        fwrite($pipes[0], $json_data);
        fclose($pipes[0]);

        // Get output and error
        $result = stream_get_contents($pipes[1]);
        fclose($pipes[1]);

        $error = stream_get_contents($pipes[2]);
        fclose($pipes[2]);

        proc_close($process);

        if (!empty($error)) {
            echo "<p style='color:red;'><strong>Error:</strong><br>" . nl2br($error) . "</p>";
        } else {
            $clean_result = trim($result);
            $prediction_result = ($clean_result === "1" || strtolower($clean_result) === "yes") ? "Yes" : "No";
            $predicted = true;
        }
    } else {
        echo "<p>Failed to start the Python process.</p>";
    }
}
?>

<h2>Change features to check attrition</h2>
<form method="POST">
    <ul style="list-style: none; padding: 0;">
        <?php
        foreach ($changable_features as $feature) {
            $value = $row[$feature];
            echo "<li><strong>" . htmlspecialchars($feature) . ":</strong> ";

            if (in_array($feature, $changable_features)) {
                if ($feature === "OverTime") {
                    echo "<select name='$feature'>
                        <option value='Yes'" . ($value === 'Yes' ? ' selected' : '') . ">Yes</option>
                        <option value='No'" . ($value === 'No' ? ' selected' : '') . ">No</option>
                    </select>";
                } elseif ($feature === "BusinessTravel") {
                    echo "<select name='$feature'>";
                    $options = ['No Travel', 'Some Travel', 'Frequent Traveller'];
                    foreach ($options as $opt) {
                        $sel = $value === $opt ? 'selected' : '';
                        echo "<option value='$opt' $sel>$opt</option>";
                    }
                    echo "</select>";
                } elseif ($feature === "StockOptionLevel") {
                    echo "<select name='$feature'>";
                    for ($i = 0; $i <= 3; $i++) {
                        $sel = $value == $i ? 'selected' : '';
                        echo "<option value='$i' $sel>$i</option>";
                    }
                    echo "</select>";
                } elseif ($feature === "Salary") {
                    echo "<input type='number' name='$feature' value='" . htmlspecialchars($value) . "'>";
                }
            } else {
                echo htmlspecialchars($value);
            }

            echo "</li>";
        }
        ?>
    </ul>
    <br>
    <button type="submit" name="predict">Check Attrition</button>
</form>

<!-- Attrition Status Section -->
<div style="margin-top: 20px; padding: 10px; border-top: 1px solid #ccc;">
    <p>
        <strong>Current Attrition Status:</strong>
        <span style="color: <?= $row['Attrition'] === 'Yes' ? 'red' : 'green' ?>;">
            <?= htmlspecialchars($row['Attrition']) ?>
        </span>
    </p>

    <?php if ($predicted): ?>
        <p>
            <strong>Predicted Attrition:</strong>
            <span style="color: <?= $prediction_result === 'Yes' ? 'red' : 'green' ?>;">
                <?= htmlspecialchars($prediction_result) ?>
            </span>
        </p>
    <?php endif; ?>
</div>