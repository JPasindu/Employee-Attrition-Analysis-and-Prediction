<?php
$host = "localhost";
$username = "root";
$password = "";
$database = "Atlas Lab";
$socket = "/Applications/XAMPP/xamppfiles/var/mysql/mysql.sock";

$conn = mysqli_connect($host, $username, $password, $database, null, $socket);

if (!$conn) {
    die("Connection failed: " . mysqli_connect_error());
}

?>