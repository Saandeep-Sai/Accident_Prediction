# Road Accident Prediction System

A Django-based web application that predicts road accidents using machine learning techniques and data mining.

## Features

- User registration and authentication
- Road accident data analysis
- Machine learning prediction models
- Data visualization and reporting
- Admin panel for data management

## Technology Stack

- **Backend**: Django (Python)
- **Database**: MySQL
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: Python libraries for data analysis

## Setup Instructions

### Option 1: WAMP Server Setup (Recommended for Windows)

1. **Download and Install WAMP Server**
   - Download from: https://www.wampserver.com/en/
   - Install WAMP64 (64-bit version recommended)
   - Start WAMP server (icon should be green in system tray)

2. **Clone the repository**
   ```bash
   git clone https://github.com/Saandeep-Sai/Accident_Prediction.git
   ```

3. **Move project to WAMP directory**
   - Copy the project folder to: `C:\wamp64\www\`
   - Final path should be: `C:\wamp64\www\Accident_Prediction\`

4. **Install Python dependencies**
   ```bash
   cd C:\wamp64\www\Accident_Prediction
   pip install django mysqlclient
   ```

5. **Setup MySQL Database**
   - Open phpMyAdmin: `http://localhost/phpmyadmin`
   - Login with username: `root`, password: (empty)
   - Import database: Go to Import tab → Choose file → Select `Database/a_road_accident_prediction.sql`
   - Click "Go" to import

6. **Verify Database Configuration**
   - Database settings are already configured in `a_road_accident_prediction/settings.py`:
     - Database: `a_road_accident_prediction`
     - User: `root`
     - Password: (empty)
     - Host: `127.0.0.1`
     - Port: `3306`

7. **Run Django Application**
   ```bash
   cd a_road_accident_prediction
   python manage.py runserver
   ```

8. **Access the Application**
   - Open browser and go to: `http://127.0.0.1:8000`
   - Register new users or login with existing credentials

9. **Monitor Database (Optional)**
   - Use phpMyAdmin to check if data is being saved
   - Check tables: `remote_user_clientregister_model`, `remote_user_road_accident_prediction`

### Option 2: Standard Django Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Saandeep-Sai/Accident_Prediction.git
   ```

2. **Install dependencies**
   ```bash
   pip install django mysqlclient
   ```

3. **Setup MySQL Database**
   - Create database: `a_road_accident_prediction`
   - Import: `mysql -u root -p < Database/a_road_accident_prediction.sql`

4. **Run migrations**
   ```bash
   python manage.py migrate
   ```

5. **Start development server**
   ```bash
   python manage.py runserver
   ```

## Project Structure

- `a_road_accident_prediction/` - Main Django project
- `Remote_User/` - User management app
- `Service_Provider/` - Service provider functionality
- `Template/` - HTML templates and static files
- `Database/` - SQL database dump
- `Road_Accidents.csv` - Dataset for analysis

## Database Tables

- `remote_user_clientregister_model` - User registrations
- `remote_user_road_accident_prediction` - Prediction data
- `remote_user_detection_accuracy` - Model accuracy metrics
- `remote_user_detection_ratio` - Detection ratios

## Application URLs

- **Home Page**: `http://127.0.0.1:8000`
- **User Registration**: `http://127.0.0.1:8000/register`
- **User Login**: `http://127.0.0.1:8000/login`
- **Admin Panel**: `http://127.0.0.1:8000/admin`
- **phpMyAdmin**: `http://localhost/phpmyadmin` (WAMP setup only)

## Troubleshooting

### WAMP Server Issues
- **WAMP icon is orange/red**: Restart all services from WAMP menu
- **Port 80 conflict**: Change Apache port in WAMP settings
- **MySQL not starting**: Check if another MySQL service is running
- **phpMyAdmin not accessible**: Ensure Apache is running on port 80

### Django Issues
- **Module not found**: Install missing dependencies with `pip install`
- **Database connection error**: Verify MySQL is running and credentials are correct
- **Port 8000 in use**: Use different port: `python manage.py runserver 8001`

### Database Issues
- **Import failed**: Check MySQL version compatibility
- **No data showing**: Verify database name and table structure
- **Connection refused**: Ensure MySQL service is running

## System Requirements

- **OS**: Windows 10/11 (for WAMP setup)
- **Python**: 3.7 or higher
- **MySQL**: 5.7 or higher
- **RAM**: Minimum 4GB
- **Storage**: 2GB free space

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for educational purposes.