<?php
include '../connection.php';

$state = trim($_GET['state'] ?? '');
$department = trim($_GET['department'] ?? '');
$jobrole = trim($_GET['jobrole'] ?? '');
$attrition = trim($_GET['attrition'] ?? '');

function buildWhereClause($conn, $filters) {
    $clauses = [];
    foreach ($filters as $field => $value) {
        if (!empty($value)) {
            $escapedValue = mysqli_real_escape_string($conn, $value);
            $clauses[] = "$field = '$escapedValue'";
        }
    }
    return !empty($clauses) ? " WHERE " . implode(" AND ", $clauses) : "";
}

$whereClause = buildWhereClause($conn, [
    'State' => $state,
    'Department' => $department,
    'JobRole' => $jobrole,
    'Attrition' => $attrition
]);

// Total Employees
$totalSql = "SELECT COUNT(*) as count FROM EmployeeData $whereClause";
$totalCount = mysqli_fetch_assoc(mysqli_query($conn, $totalSql))['count'];

// Employees by State
$stateList = "";
$res = mysqli_query($conn, "SELECT State, COUNT(*) as count FROM EmployeeData $whereClause GROUP BY State ORDER BY count DESC");
while ($row = mysqli_fetch_assoc($res)) {
    $stateList .= "<li>{$row['State']}: {$row['count']}</li>";
}

// Department
$deptList = "";
$res = mysqli_query($conn, "SELECT Department, COUNT(*) as count FROM EmployeeData $whereClause GROUP BY Department ORDER BY count DESC");
while ($row = mysqli_fetch_assoc($res)) {
    $deptList .= "<li>{$row['Department']}: {$row['count']}</li>";
}

// Job Role
$roleList = "";
$res = mysqli_query($conn, "SELECT JobRole, COUNT(*) as count FROM EmployeeData $whereClause GROUP BY JobRole ORDER BY count DESC");
while ($row = mysqli_fetch_assoc($res)) {
    $roleList .= "<li>{$row['JobRole']}: {$row['count']}</li>";
}

// Attrition
$attrList = "";
$res = mysqli_query($conn, "SELECT Attrition, COUNT(*) as count FROM EmployeeData $whereClause GROUP BY Attrition ORDER BY Attrition DESC");
while ($row = mysqli_fetch_assoc($res)) {
    $attrList .= "<li>{$row['Attrition']}: {$row['count']}</li>";
}

echo <<<HTML
    <h3>Total Employees</h3>
    <p><strong>$totalCount</strong></p>

    <h3>By State</h3>
    <ul>$stateList</ul>

    <h3>By Department</h3>
    <ul>$deptList</ul>

    <h3>By Job Role</h3>
    <ul>$roleList</ul>

    <h3>By Attrition</h3>
    <ul>$attrList</ul>
HTML;

mysqli_close($conn);
?>
