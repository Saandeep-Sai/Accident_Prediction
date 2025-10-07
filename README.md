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

1. Clone the repository:
   ```bash
   git clone https://github.com/Saandeep-Sai/Accident_Prediction.git
   ```

2. Install dependencies:
   ```bash
   pip install django mysqlclient
   ```

3. Configure database in `settings.py`:
   - Database: `a_road_accident_prediction`
   - User: `root`
   - Password: (empty for local development)

4. Import the database:
   ```bash
   mysql -u root -p < Database/a_road_accident_prediction.sql
   ```

5. Run migrations:
   ```bash
   python manage.py migrate
   ```

6. Start the development server:
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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for educational purposes.