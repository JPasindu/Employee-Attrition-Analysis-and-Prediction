<?php include('../navbar.html'); ?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <link rel="stylesheet" href="styles/style_form.css">
</head>
<body></body>

<?php

ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

include '../connection.php';

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // Initialize variables from POST
    $EmployeeID = $_POST["EmployeeID"] ?? "";
    $FirstName = $_POST["FirstName"] ?? "";
    $LastName = $_POST["LastName"] ?? "";
    $Gender = $_POST["Gender"] ?? "";
    $Age = $_POST["Age"] ?? "";
    $BusinessTravel = $_POST["BusinessTravel"] ?? "";
    $Department = $_POST["Department"] ?? "";
    $DistanceFromHome_KM = $_POST["DistanceFromHome_KM"] ?? "";
    $State = $_POST["State"] ?? "";
    $Ethnicity = $_POST["Ethnicity"] ?? "";
    $Education = $_POST["Education"] ?? "";
    $EducationField = $_POST["EducationField"] ?? "";
    $JobRole = $_POST["JobRole"] ?? "";
    $MaritalStatus = $_POST["MaritalStatus"] ?? "";
    $Salary = $_POST["Salary"] ?? "";
    $StockOptionLevel = $_POST["StockOptionLevel"] ?? "";
    $OverTime = $_POST["OverTime"] ?? "";
    $HireDate = $_POST["HireDate"] ?? "";
    $YearsAtCompany = $_POST["YearsAtCompany"] ?? "";
    $YearsInMostRecentRole = $_POST["YearsInMostRecentRole"] ?? "";
    $YearsSinceLastPromotion = $_POST["YearsSinceLastPromotion"] ?? "";
    $YearsWithCurrManager = $_POST["YearsWithCurrManager"] ?? "";
    $ReviewDate = $_POST["ReviewDate"] ?? "";
    $EnvironmentSatisfaction = $_POST["EnvironmentSatisfaction"] ?? "";
    $JobSatisfaction = $_POST["JobSatisfaction"] ?? "";
    $RelationshipSatisfaction = $_POST["RelationshipSatisfaction"] ?? "";
    $TrainingOpportunitiesWithinYear = $_POST["TrainingOpportunitiesWithinYear"] ?? "";
    $TrainingOpportunitiesTaken = $_POST["TrainingOpportunitiesTaken"] ?? "";
    $WorkLifeBalance = $_POST["WorkLifeBalance"] ?? "";
    $SelfRating = $_POST["SelfRating"] ?? "";
    $ManagerRating = $_POST["ManagerRating"] ?? "";


    // convert the date to proper format
    $HireDate = date('Y-m-d', strtotime($HireDate));
    $ReviewDate = date('Y-m-d', strtotime($ReviewDate));

    // Create associative array for JSON
    $data = array(
        "BusinessTravel" => $BusinessTravel,
        "MaritalStatus" => $MaritalStatus,
        "StockOptionLevel" => $StockOptionLevel,
        "OverTime" => $OverTime,
        "Age"=>$Age,
        "Salary" => $Salary,
        "YearsSinceLastPromotion" => $YearsSinceLastPromotion,
    );

    $json_data = json_encode($data);

    // Set Python command
    // $command = "/Users/pasindumadusanka/myenv/bin/python predict_attrition.py";
    include "../python_connection.php";
    $command = $path."predict_attrition.py";

    $descriptorspec = [
        0 => ["pipe", "r"],  // stdin
        1 => ["pipe", "w"],  // stdout
        2 => ["pipe", "w"]   // stderr
    ];
    
    $process = proc_open($command, $descriptorspec, $pipes);

    if (is_resource($process)) {
        fwrite($pipes[0], $json_data);
        fclose($pipes[0]);

        $result = stream_get_contents($pipes[1]);
        fclose($pipes[1]);

        $error = stream_get_contents($pipes[2]);
        fclose($pipes[2]);

        proc_close($process);

        if (!empty($error)) {
            echo "<p style='color:red;'><strong>Error:</strong><br>" . nl2br($error) . "</p>";
        } else {
            $clean_result = trim($result);
            $Attrition = ($clean_result === "1") ? "Yes" : "No";
        }
    } else {
        echo "<p>Failed to start the Python process.</p>";
    }

    //-----------------------------------
    // --- Insert into database ---
    $sql = "INSERT INTO EmployeeData (
    EmployeeID, FirstName, LastName, Gender, Age, BusinessTravel, Department,
    DistanceFromHome_KM, State, Ethnicity, Education, EducationField, JobRole,
    MaritalStatus, Salary, StockOptionLevel, OverTime, HireDate,
    Attrition,                         
    YearsAtCompany, YearsInMostRecentRole, YearsSinceLastPromotion, YearsWithCurrManager,
    ReviewDate, EnvironmentSatisfaction, JobSatisfaction, RelationshipSatisfaction,
    TrainingOpportunitiesWithinYear, TrainingOpportunitiesTaken, WorkLifeBalance,
    SelfRating, ManagerRating
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ";

    $stmt = mysqli_prepare($conn, $sql);
    if (!$stmt) {
        die("<p>❌ SQL Prepare Error: " . mysqli_error($conn) . "</p>");
    }

    // Bind parameters
    $bindResult = mysqli_stmt_bind_param(
        $stmt, 
        "ssssississssssissssiiiisiiiiiiii",
        $EmployeeID, $FirstName, $LastName, $Gender, $Age, $BusinessTravel, $Department,
        $DistanceFromHome_KM, $State, $Ethnicity, $Education, $EducationField, $JobRole, $MaritalStatus,
        $Salary, $StockOptionLevel, $OverTime, $HireDate,
        $Attrition,
        $YearsAtCompany, $YearsInMostRecentRole, $YearsSinceLastPromotion, $YearsWithCurrManager,
        $ReviewDate, $EnvironmentSatisfaction, $JobSatisfaction, $RelationshipSatisfaction,
        $TrainingOpportunitiesWithinYear, $TrainingOpportunitiesTaken, $WorkLifeBalance,
        $SelfRating, $ManagerRating
    );

    if (!$bindResult) {
        die("<p>❌ Bind Error: " . mysqli_stmt_error($stmt) . "</p>");
    }
    
    // Execute query
    if (!mysqli_stmt_execute($stmt)) {
        die("<p>❌ Execute Error: " . mysqli_stmt_error($stmt) . "</p>");
    }

    // Success message with prediction result
    echo "<div style='padding: 20px; background-color: #f8f9fa; border-radius: 5px; margin: 20px 0;'>";
    echo "<h3>Employee Record Submitted Successfully</h3>";
    echo "<p><strong>Attrition Prediction:</strong> " . ($Attrition == 'Yes' ? 
         '<span style="color:red;">High Risk (Predicted to Leave)</span>' : 
         '<span style="color:green;">Low Risk (Predicted to Stay)</span>') . "</p>";
    echo "<p><a href='form/form.php'>Submit Another Record</a></p>";
    echo "</div>";

    // Clean up
    mysqli_stmt_close($stmt);

}

?>

<!-- HTML Form -->
<h2>Insert Full Employee Record</h2>
<form method="POST" action="">
    <!-- Numerical Inputs -->

    <label>EmployeeID:</label><br>
    <input type='text' name='EmployeeID' title="Format: XXXX-XXXX (hex characters)" required><br><br>

    <!-- Text Inputs -->
    <label>FirstName:</label><br>
    <input type='text' name='FirstName' required><br><br>

    <label>LastName:</label><br>
    <input type='text' name='LastName' required><br><br>

    <!-- Categorical Dropdowns -->
    <label>Gender:</label><br>
    <select name='Gender' required>
        <option value='Male'>Male</option>
        <option value='Female'>Female</option>
        <option value='Non-Binary'>Non-Binary</option>
        <option value='Prefer Not To Say'>Prefer Not To Say</option>
    </select><br><br>

    <label>Age:</label><br>
    <input type='number' name='Age' min='18' max='65' required><br><br>

    <label>BusinessTravel:</label><br>
    <select name='BusinessTravel' required>
        <option value='Frequent Traveller'>Frequent Traveller</option>
        <option value='Some Travel'>Some Travel</option>
        <option value='No Travel'>No Travel</option>
    </select><br><br>

    <label>Department:</label><br>
    <select name='Department' required>
        <option value='Sales'>Sales</option>
        <option value='Technology'>Technology</option>
        <option value='Human Resources'>Human Resources</option>
    </select><br><br>

    <label>DistanceFromHome_KM:</label><br>
    <input type='number' name='DistanceFromHome_KM' min='0' max='100' required><br><br>

    <label>State:</label><br>
    <select name='State' required>
        <option value='CA'>CA</option>
        <option value='NY'>NY</option>
        <option value='IL'>IL</option>
    </select><br><br>

    <label>Ethnicity:</label><br>
    <select name='Ethnicity' required>
        <option value='White'>White</option>
        <option value='Black or African American'>Black or African American</option>
        <option value='Asian or Asian American'>Asian or Asian American</option>
        <option value='American Indian or Alaska Native'>American Indian or Alaska Native</option>
        <option value='Mixed or multiple ethnic groups'>Mixed or multiple ethnic groups</option>
        <option value='Native Hawaiian'>Native Hawaiian</option>
        <option value='Other'>Other</option>
    </select><br><br>

    <label>Education:</label><br>
    <input type='number' name='Education' min='1' max='5' required><br><br>
    EducationLevelID	EducationLevel
	
    <label>Education:</label><br>
    <select name='EducationField' required>
        <option value='1'>No Formal Qualifications</option>
        <option value='2'>High School</option>
        <option value='3'>Bachelors</option>
        <option value='4'>Masters</option>
        <option value='5'>Doctorate</option>
    </select><br><br>

    <label>EducationField:</label><br>
    <select name='EducationField' required>
        <option value='Marketing'>Marketing</option>
        <option value='Computer Science'>Computer Science</option>
        <option value='Business Studies'>Business Studies</option>
        <option value='Information Systems'>Information Systems</option>
        <option value='Technical Degree'>Technical Degree</option>
        <option value='Economics'>Economics</option>
        <option value='Other'>Other</option>
        <option value='Human Resources'>Human Resources</option>
    </select><br><br>

    <label>JobRole:</label><br>
    <select name='JobRole' required>
        <option value='Sales Executive'>Sales Executive</option>
        <option value='Machine Learning Engineer'>Machine Learning Engineer</option>
        <option value='Software Engineer'>Software Engineer</option>
        <option value='Data Scientist'>Data Scientist</option>
        <option value='Sales Representative'>Sales Representative</option>
        <option value='Manager'>Manager</option>
        <option value='Analytics Manager'>Analytics Manager</option>
        <option value='Senior Software Engineer'>Senior Software Engineer</option>
        <option value='Engineering Manager'>Engineering Manager</option>
        <option value='HR Executive'>HR Executive</option>
        <option value='Recruiter'>Recruiter</option>
        <option value='HR Manager'>HR Manager</option>
    </select><br><br>

    <label>MaritalStatus:</label><br>
    <select name='MaritalStatus' required>
        <option value='Married'>Married</option>
        <option value='Single'>Single</option>
        <option value='Divorced'>Divorced</option>
    </select><br><br>

    <!-- Continue with remaining fields... -->
    <!-- Add similar dropdowns for other categorical fields -->

    <label>Salary:</label><br>
    <input type='number' name='Salary' min='20000' max='500000' required><br><br>

    <label>StockOptionLevel:</label><br>
    <input type='number' name='StockOptionLevel' min='0' max='3' required><br><br>

    <label>OverTime:</label><br>
    <select name='OverTime' required>
        <option value='Yes'>Yes</option>
        <option value='No'>No</option>
    </select><br><br>

    <label>HireDate:</label><br>
    <input type='date' name='HireDate' required><br><br>

    <!--
    <label>Attrition:</label><br>
    <select name='Attrition' required>
        <option value='Yes'>Yes</option>
        <option value='No'>No</option>
    </select><br><br>
    -->

    <!-- Add remaining numerical/date fields... -->
    <!-- Years-related Fields -->
    <label>YearsAtCompany:</label><br>
    <input type="number" name="YearsAtCompany" min="0" max="50" required><br><br>

    <label>YearsInMostRecentRole:</label><br>
    <input type="number" name="YearsInMostRecentRole" min="0" max="20" required><br><br>

    <label>YearsSinceLastPromotion:</label><br>
    <input type="number" name="YearsSinceLastPromotion" min="0" max="20" required><br><br>

    <label>YearsWithCurrManager:</label><br>
    <input type="number" name="YearsWithCurrManager" min="0" max="20" required><br><br>

    <!-- Performance ID 
    <label>PerformanceID:</label><br>
    <input type="text" name="PerformanceID" pattern="PR\d{4}" title="Performance ID format: PR followed by 4 digits" required><br><br>//
    -->

    <!-- Review Date -->
    <label>ReviewDate:</label><br>
    <input type="date" name="ReviewDate" required><br><br>

    <!-- Ordinal Scales with Labels -->
    <label>EnvironmentSatisfaction:</label><br>
    <select name="EnvironmentSatisfaction" required>
        <option value="1">1 - Very Dissatisfied</option>
        <option value="2">2 - Dissatisfied</option>
        <option value="3">3 - Neutral</option>
        <option value="4">4 - Satisfied</option>
        <option value="5">5 - Very Satisfied</option>
    </select><br><br>

    <label>JobSatisfaction:</label><br>
    <select name="JobSatisfaction" required>
        <option value="1">1 - Very Dissatisfied</option>
        <option value="2">2 - Dissatisfied</option>
        <option value="3">3 - Neutral</option>
        <option value="4">4 - Satisfied</option>
        <option value="5">5 - Very Satisfied</option>
    </select><br><br>

    <label>RelationshipSatisfaction:</label><br>
    <select name="RelationshipSatisfaction" required>
        <option value="1">1 - Very Poor</option>
        <option value="2">2 - Poor</option>
        <option value="3">3 - Average</option>
        <option value="4">4 - Good</option>
        <option value="5">5 - Excellent</option>
    </select><br><br>

    <label>TrainingOpportunitiesWithinYear:</label><br>
    <select name="TrainingOpportunitiesWithinYear" required>
        <option value="1">1 - No Opportunities</option>
        <option value="2">2 - Limited Opportunities</option>
        <option value="3">3 - Adequate Opportunities</option>
    </select><br><br>

    <label>TrainingOpportunitiesTaken:</label><br>
    <input type="number" name="TrainingOpportunitiesTaken" min="0" max="3" required><br><br>

    <label>WorkLifeBalance:</label><br>
    <select name="WorkLifeBalance" required>
        <option value="1">1 - Poor Balance</option>
        <option value="2">2 - Below Average</option>
        <option value="3">3 - Balanced</option>
        <option value="4">4 - Good Balance</option>
        <option value="5">5 - Excellent Balance</option>
    </select><br><br>

    <label>SelfRating:</label><br>
    <select name="SelfRating" required>
        <option value="1">1 - Needs Improvement</option>
        <option value="2">2 - Developing</option>
        <option value="3">3 - Competent</option>
        <option value="4">4 - Strong Performer</option>
        <option value="5">5 - Exceptional</option>
    </select><br><br>

    <label>ManagerRating:</label><br>
    <select name="ManagerRating" required>
        <option value="1">1 - Below Expectations</option>
        <option value="2">2 - Approaching Expectations</option>
        <option value="3">3 - Meets Expectations</option>
        <option value="4">4 - Exceeds Expectations</option>
        <option value="5">5 - Outstanding</option>
    </select><br><br>

    <input type="submit" value="Insert Record">
</form>