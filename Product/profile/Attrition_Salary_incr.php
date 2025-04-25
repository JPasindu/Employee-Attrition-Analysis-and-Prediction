<?php   

// Convert to JSON
$json_data = json_encode($input_data);

// Python script path
// $command = "/Users/pasindumadusanka/myenv/bin/python3 Salary_pred_incr.py";
include "../python_connection.php";

$command = $path."Salary_pred_incr.py";

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
        $prediction_range = $clean_result;
        $predicted = true;
    }
} else {
    echo "<p>Failed to start the Python process.</p>";
}
