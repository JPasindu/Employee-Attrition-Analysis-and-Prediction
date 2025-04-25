
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Atlas Labs - Admin Dashboard</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --primary: #2563eb;
            --dark: #1e3a8a;
            --light: #eff6ff;
            --text: #1f2937;
            --gray: #6b7280;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f9fafb;
            color: var(--text);
        }
        
        .container {
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .dashboard-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
        }
        
        .dashboard-title h1 {
            font-size: 1.75rem;
            font-weight: 600;
            margin: 0;
        }
        
        .dashboard-title p {
            color: var(--gray);
            margin: 0.25rem 0 0;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .stat-card {
            background: white;
            border-radius: 0.5rem;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        
        .stat-card:hover {
            transform: translateY(-2px);
        }
        
        .stat-card h3 {
            font-size: 0.875rem;
            color: var(--gray);
            margin: 0 0 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .stat-card .value {
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0;
            color: var(--primary);
        }
        
        .stat-card .change {
            display: flex;
            align-items: center;
            font-size: 0.875rem;
            margin-top: 0.5rem;
        }
        
        .change.positive {
            color: #10b981;
        }
        
        .change.negative {
            color: #ef4444;
        }
        
        .content-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1.5rem;
        }
        
        .panel {
            background: white;
            border-radius: 0.5rem;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }
        
        .panel-header h2 {
            font-size: 1.25rem;
            margin: 0;
        }
        
        .btn {
            padding: 0.5rem 1rem;
            border-radius: 0.375rem;
            font-weight: 500;
            cursor: pointer;
            border: none;
            transition: background-color 0.2s;
        }
        
        .btn-primary {
            background-color: var(--primary);
            color: white;
        }
        
        .btn-primary:hover {
            background-color: var(--dark);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th {
            text-align: left;
            padding: 0.75rem 1rem;
            background-color: #f3f4f6;
            color: var(--gray);
            font-weight: 500;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }
        
        td {
            padding: 1rem;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .employee-avatar {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background-color: #e5e7eb;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        
        .avatar img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .status {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        
        .status-active {
            background-color: #dcfce7;
            color: #16a34a;
        }
        
        .status-inactive {
            background-color: #fee2e2;
            color: #dc2626;
        }
        
        .activity-item {
            display: flex;
            gap: 1rem;
            padding: 0.75rem 0;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .activity-item:last-child {
            border-bottom: none;
        }
        
        .activity-icon {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background-color: var(--light);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--primary);
        }
        
        .activity-content p {
            margin: 0;
        }
        
        .activity-time {
            font-size: 0.75rem;
            color: var(--gray);
            margin-top: 0.25rem;
        }
    </style>
</head>
<body>
    <!-- Include top bar -->
    <?php include('../navbar.html'); ?>
    <?php
        include('connection.php');

        $employeeCount = 0;
        $sqlCount = "SELECT COUNT(*) as total FROM EmployeeData";
        $resultCount = $conn->query($sqlCount);

        if ($resultCount && $row = $resultCount->fetch_assoc()) {
            $employeeCount = $row['total'];
        }
    ?>

    <div class="container">
        <!-- Dashboard Header -->
        <div class="dashboard-header">
            <div class="dashboard-title">
                <h1>Dashboard Overview</h1>
                <p>Welcome back, Admin. Here's what's happening with your company today.</p>
            </div>
            <div>
            <div>
                <a href="form/form.php" class="btn btn-primary">
                    <i class="fas fa-plus"></i> Add Employee
                </a>
            </div>
            </div>
        </div>

        <!-- Statistics Grid -->
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Employees</h3>
                    <p class="value"><?php echo $employeeCount; ?></p>
                <div class="change positive">
                    <i class="fas fa-arrow-up"></i> 12% from last month
                </div>
            </div>
            <div class="stat-card">
                <h3>Active Projects</h3>
                <p class="value">18</p>
                <div class="change positive">
                    <i class="fas fa-arrow-up"></i> 3 new this week
                </div>
            </div>
            <div class="stat-card">
                <h3>Pending Tasks</h3>
                <p class="value">27</p>
                <div class="change negative">
                    <i class="fas fa-arrow-down"></i> 5 overdue
                </div>
            </div>
            <div class="stat-card">
                <h3>Upcoming Events</h3>
                <p class="value">4</p>
                <div class="change">
                    <i class="fas fa-calendar"></i> View calendar
                </div>
            </div>
        </div>

        <!-- Main Content Grid -->
        <div class="content-grid">
            <!-- Recent Employees Panel -->
            <div class="panel">
                <div class="panel-header">
                    <h2>Recent Employees</h2>
                    <a href="view_data/view_dataset.php" class="panel-link">View All</a>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>State</th>
                            <th>Position</th>
                            <th>Department</th>
                            <th>Attrition</th>
                            <th>Action</th>
    
                        </tr>
                    </thead>
                    <tbody>
                    <?php
                        include('connection.php');

                        // Get the last 5 rows (newest entries based on EmployeeID)
                        $sql = "SELECT EmployeeID, FirstName, State, Department, JobRole, Age, Gender, Attrition FROM EmployeeData";

                        $result = $conn->query($sql);

                        if ($result && $result->num_rows > 0):
                            $rows = [];
                            while ($row = $result->fetch_assoc() ) {
                                $rows[] = $row;
                            }
                            $rows = array_reverse($rows); // Show last row first
                            $rows = array_slice($rows, 0, 5);
                            foreach ($rows as $row):
                        ?>
                        <tr>
                            <td>
                                <div class="employee-avatar">
                                    <div class="avatar"><i class="fas fa-user"></i></div>
                                    <span><?php echo htmlspecialchars($row['FirstName']); ?></span>
                                </div>
                            </td>
                            <td><?php echo htmlspecialchars($row['State']); ?></td>
                            <td><?php echo htmlspecialchars($row['JobRole']); ?></td>
                            <td><?php echo htmlspecialchars($row['Department']); ?></td>
                            <td><?php echo htmlspecialchars($row['Attrition']); ?></td>
                            <td><a href="view_data/view_dataset.php" class="action-link">View</a></td>
                        </tr>
                        <?php
                            endforeach;
                        else:
                        ?>
                        <tr><td colspan="8">No employee data found.</td></tr>
                        <?php endif; ?>


                    </tbody>
                </table>
            </div>

            <!-- Recent Activity Panel -->
            <div class="panel">
                <div class="panel-header">
                    <h2>Recent Activity</h2>
                </div>
                <div class="activity-list">
                    <div class="activity-item">
                        <div class="activity-icon"><i class="fas fa-user-plus"></i></div>
                        <div class="activity-content">
                            <p><strong>New employee</strong> - Robert Taylor was added to Engineering team</p>
                            <p class="activity-time">2 hours ago</p>
                        </div>
                    </div>
                    <div class="activity-item">
                        <div class="activity-icon"><i class="fas fa-tasks"></i></div>
                        <div class="activity-content">
                            <p><strong>Project update</strong> - Atlas Platform v2.3 was deployed</p>
                            <p class="activity-time">5 hours ago</p>
                        </div>
                    </div>
                    <div class="activity-item">
                        <div class="activity-icon"><i class="fas fa-file-alt"></i></div>
                        <div class="activity-content">
                            <p><strong>Report generated</strong> - Q3 Financial Report is ready</p>
                            <p class="activity-time">Yesterday</p>
                        </div>
                    </div>
                    <div class="activity-item">
                        <div class="activity-icon"><i class="fas fa-calendar-check"></i></div>
                        <div class="activity-content">
                            <p><strong>Event reminder</strong> - All-hands meeting tomorrow at 10 AM</p>
                            <p class="activity-time">Yesterday</p>
                        </div>
                    </div>
                </div>
            </div>
        </div> <!-- End content-grid -->
    </div> <!-- End container -->
</body>
</html>
