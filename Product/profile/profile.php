<?php include('../navbar.html'); ?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Employee Profile</title>
    <link rel="stylesheet" href="styles/style_panel.css?v=<?php echo time(); ?>">
    <link rel="stylesheet" href="styles/style_profile.css?v=<?php echo time(); ?>">

</head>
<body>

<?php

include '../connection.php';

$empId = $_GET['id'] ?? '';

if (empty($empId)) {
    echo "<p>Invalid Employee ID</p>";
    exit;
}

$stmt = mysqli_prepare($conn, "SELECT * FROM EmployeeData WHERE EmployeeID = ?");
mysqli_stmt_bind_param($stmt, 's', $empId);
mysqli_stmt_execute($stmt);
$result = mysqli_stmt_get_result($stmt);

if (mysqli_num_rows($result) === 0) {
    echo "<p>Employee not found</p>";
    exit;
}

$row = mysqli_fetch_assoc($result);

?>
<br>
<h2>Employee Profile: <?php echo htmlspecialchars($row['EmployeeID']); ?></h2>

<div class="container">
    <!-- Full Profile Table -->
    <div class="profile-table">
        <table border="1">
            <?php
            foreach ($row as $key => $value) {
                echo "<tr>";
                echo "<th>" . htmlspecialchars($key) . "</th>";
                echo "<td>" . htmlspecialchars($value) . "</td>";
                echo "</tr>";
            }
            ?>
        </table>
        <a href="../view_data/view_dataset.php">← Back</a>
    </div>

    <!-- Side panels with Employee Panel on top -->
    <div class="side-panels">
         <div class="panel-box">
            <?php include 'Attrition.php'; ?>
        </div>
        <div class="panel-box">
            <?php include "Employee_panel.php"; ?>
        </div>
        <div class="panel-box">
            <?php include "promotion.php"; ?>
        </div>
        
    </div>
</div>


</body>
</html>
