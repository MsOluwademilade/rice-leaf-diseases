"""
Flask Documentation:     https://flask.palletsprojects.com/
Jinja2 Documentation:    https://jinja.palletsprojects.com/
Werkzeug Documentation:  https://werkzeug.palletsprojects.com/
This file contains the routes for your application.
"""

import pandas as pd
from app import app
import tensorflow as tf
from flask import flash, render_template, request

rice_model = tf.keras.models.load_model('app/models/vgg16_rice_model.h5')

###
# Routing for your application.
###


@app.route("/")
def home():
    """Render website's home page."""
    return render_template("home.html")


@app.route("/about/")
def about():
    """Render the website's about page."""
    return render_template("about.html", name="Mary Jane")


###
# API routes, should return html
###


@app.route("/predict", methods=["POST"])
def predict():
    # collect data from form
    Year = request.form["Year"]
    average_rain_fall_mm_per_year = request.form["average_rain_fall_mm_per_year"]
    Pesticides_Value_Tonnes = request.form["Pesticides_Value_Tonnes"]
    avg_temp = request.form["avg_temp"]
    Land_Area_ha = request.form["Land_Area_ha"]
    avg_population = request.form["avg_population"]

    data = {
        "Year": Year,
        "average_rain_fall_mm_per_year": average_rain_fall_mm_per_year,
        "Pesticides_Value_Tonnes": Pesticides_Value_Tonnes,
        "avg_temp": avg_temp,
        "Land_Area_ha": Land_Area_ha,
        "avg_population": avg_population,
    }

    prediction = rice_model.predict(data)

    return render_template(
        "predict.html",
        result=prediction,
    )


###
# The functions below should be applicable to all Flask apps.
###


# Display Flask WTF errors as Flash messages
def flash_errors(form):
    for field, errors in form.errors.items():
        for error in errors:
            flash(
                "Error in the %s field - %s" % (getattr(form, field).label.text, error),
                "danger",
            )


@app.route("/<file_name>.txt")
def send_text_file(file_name):
    """Send your static text file."""
    file_dot_text = file_name + ".txt"
    return app.send_static_file(file_dot_text)


@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also tell the browser not to cache the rendered page. If we wanted
    to we could change max-age to 600 seconds which would be 10 minutes.
    """
    response.headers["X-UA-Compatible"] = "IE=Edge,chrome=1"
    response.headers["Cache-Control"] = "public, max-age=0"
    return response


@app.errorhandler(404)
def page_not_found(error):
    """Custom 404 page."""
    return render_template("404.html"), 404
