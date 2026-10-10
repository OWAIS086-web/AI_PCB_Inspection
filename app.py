import json
import os
import time
import datetime
from datetime import datetime
from datetime import datetime, timezone, timedelta
from flask_migrate import Migrate
import threading
from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    jsonify,
    flash,
    send_from_directory,
    abort,
)
from flask_sqlalchemy import SQLAlchemy
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import pytz
import base64
import os
from threading import Lock

PKT = pytz.timezone("Asia/Karachi")
app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config["CONFIG_PATH"] = "configs/config.json"
STATUS_FILE = os.path.join(BASE_DIR, "configs", "status.json")
app.config["SECRET_KEY"] = "086086086"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
TEMPLATES_PATH = "pcb_templates"
STATIC_IMAGES_PATH = "static/images"
UPLOAD_FOLDER = "detect_result"
UPLOAD_FOLDER = "static/uploads/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
os.makedirs("configs", exist_ok=True)
latest_frame_action = None
lock = Lock()
inspection_started = False


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_super_admin = db.Column(db.Boolean, default=False)
    name = db.Column(db.String(150), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    worker_type = db.Column(db.String(100), nullable=True)
    profile_image = db.Column(db.String(100), nullable=True)
    last_login = db.Column(db.DateTime, default=datetime.now)
    last_logout = db.Column(db.DateTime, nullable=True)
    total_time_spent = db.Column(db.Integer, default=0)

    def __repr__(self):
        return f"<User {self.username}>"


with app.app_context():
    db.create_all()

    super_admin = User.query.filter_by(username="manager@ai.com").first()
    if not super_admin:
        hashed_password = generate_password_hash("Admin@123", method="pbkdf2:sha256")
        super_admin = User(
            username="manager@ai.com",
            password_hash=hashed_password,
            is_super_admin=True,
        )
        db.session.add(super_admin)
        db.session.commit()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class Defect(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    image_path = db.Column(db.String(300), nullable=False)
    defect_type = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    remarks = db.Column(db.Text, nullable=True)
    panel_number = db.Column(db.String(100), nullable=True)  # New field
    pcb_number = db.Column(db.String(100), nullable=True)  # New field

    def __repr__(self):
        return f"<Defect {self.id} - {self.defect_type}>"


class PCBDefect(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    defect_data = db.Column(db.Text, nullable=False)
    details_data = db.Column(db.Text, nullable=False)
    checklist_item = db.Column(db.String(100))
    points = db.Column(db.String(20))
    checked_items = db.Column(db.Integer)
    total_items = db.Column(db.Float)
    time_taken = db.Column(db.String(50))
    defect_found = db.Column(db.String(50))

    def __init__(
        self,
        user_id,
        defect_data,
        details_data,
        checklist_item,
        points,
        checked_items,
        total_items,
        time_taken,
        defect_found,
    ):
        self.user_id = user_id
        self.defect_data = defect_data
        self.details_data = details_data
        self.checklist_item = checklist_item
        self.points = points
        self.checked_items = checked_items
        self.total_items = total_items
        self.time_taken = time_taken
        self.defect_found = defect_found


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "png",
        "jpg",
        "jpeg",
    }


os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# static page render satrt from  here
@app.route("/inspection")
@login_required
def loading_pcb():
    if current_user.is_super_admin:
        flash("You do not have access to this page.", "danger")
        return redirect(url_for("analytic"))
    return render_template("index.html")


@app.route("/history")
@login_required
def history():
    return render_template("history.html")


@app.route("/")
@login_required
def index():
    user = current_user  # current_user is the logged-in user (from Flask-Login)
    return render_template("home.html", user=user)


@app.route("/setting")
@login_required
def setting():
    return render_template("setting.html")


@app.route("/analytic")
@login_required
def analytic():
    user = current_user  # current_user is the logged-in user (from Flask-Login)
    return render_template("analytic.html", user=user)


@app.errorhandler(404)
@login_required
def page_not_found(error):
    user = current_user
    return render_template("404.html", username=user.username), 404


@app.errorhandler(500)
@login_required
def internal_server_error(error):
    user = current_user
    return render_template("500.html", username=user.username), 500


# here functionality end


# here user-mangemt is satrt from here
@app.route("/signup", methods=["GET", "POST"])
def signup():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            user.last_login = datetime.now()
            db.session.commit()
            login_user(user)
            flash("Login success!", "success")
            if user.is_super_admin:
                return redirect(url_for("analytic"))
            return redirect(url_for("index"))

        else:
            flash("Invalid credentials, please try again.", "danger")

    return render_template("login.html")


@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    if not current_user.is_super_admin:
        flash("You do not have permission to access this page.", "danger")
        return redirect(url_for("index"))

    users = User.query.all()
    return render_template("admin_dashboard.html", users=users)


@app.route("/admin/add_user", methods=["GET", "POST"])
@login_required
def add_user():
    if not current_user.is_super_admin:
        flash("You do not have permission to access this page.", "danger")
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        name = request.form["name"]
        phone = request.form["phone"]
        worker_type = request.form["worker_type"]
        image = request.files.get("profile_image")

        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image.save(filepath)
            filepath = filepath.replace("\\", "/")  # Normalize path separators
        else:
            filepath = None

        hashed_password = generate_password_hash(password, method="pbkdf2:sha256")
        new_user = User(
            username=username,
            password_hash=hashed_password,
            name=name,
            phone=phone,
            worker_type=worker_type,
            profile_image=filepath,
        )
        db.session.add(new_user)
        db.session.commit()

        flash("User added successfully!", "success")
        return redirect(url_for("admin_dashboard"))

    return render_template("add_user.html")


@app.route("/admin/update_user/<int:user_id>", methods=["GET", "POST"])
@login_required
def update_user(user_id):
    if not current_user.is_super_admin:
        flash("You do not have permission to access this page.", "danger")
        return redirect(url_for("index"))

    user = User.query.get_or_404(user_id)

    if request.method == "POST":
        user.name = request.form["name"]
        user.phone = request.form["phone"]
        user.worker_type = request.form["worker_type"]
        image = request.files.get("profile_image")

        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image.save(filepath)
            user.profile_image = filepath.replace("\\", "/")  # Normalize path separators

        db.session.commit()

        flash("User updated successfully!", "success")
        return redirect(url_for("admin_dashboard"))

    return render_template("update_user.html", user=user)


@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
@login_required
def delete_user(user_id):
    if not current_user.is_super_admin:
        flash("You do not have permission to delete users.", "danger")
        return redirect(url_for("admin_dashboard"))

    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()

    flash(f"User {user.username} has been deleted.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user = current_user

    if request.method == "POST":
        user.name = request.form["name"]
        user.phone = request.form["phone"]
        user.worker_type = request.form["worker_type"]
        image = request.files.get("profile_image")

        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image.save(filepath)
            user.profile_image = filepath.replace("\\", "/")  # Normalize path separators

        db.session.commit()
        flash("Profile updated successfully!", "success")
        return redirect(url_for("index"))

    return render_template("profile.html", user=user)


@app.route("/logout")
def logout():
    user = User.query.filter_by(id=current_user.id).first()

    if user:
        if user.last_login:
            time_spent = (datetime.now() - user.last_login).total_seconds()
            user.total_time_spent += time_spent
        user.last_logout = datetime.now()

        db.session.commit()

    logout_user()
    flash("You have been logged out!", "info")
    return redirect(url_for("login"))


# here functionality end


# here select templete page functiona start
@app.route("/select_template", methods=["POST"])
@login_required
def select_template():
    if current_user.is_super_admin:
        flash("You do not have permission to select templates.", "danger")
        return redirect(url_for("analytic"))
    template = request.form.get("templateCounter")
    config_path = "configs/config.json"

    try:
        with open(config_path, "r") as file:
            config = json.load(file)

        config["template"] = template

        with open(config_path, "w") as file:
            json.dump(config, file, indent=4)

        flash("Configuration updated successfully.", "success")
    except Exception as e:
        flash(f"Error updating configuration: {e}", "danger")

    return redirect(url_for("loading_pcb"))


# here functionality end

# now stream functionality is satrt from start


@app.route("/upload", methods=["POST"])
def upload_frame():
    global latest_frame_action
    data = request.json

    if "frame_action" not in data:
        return jsonify({"error": "Required frame not provided"}), 400

    frame_data_action = data["frame_action"]
    frame_bytes_action = base64.b64decode(frame_data_action)
    frame_path_action = os.path.join(
        "static", "received_frame_action.jpg"
    )
    with lock:
        with open(frame_path_action, "wb") as f_action:
            f_action.write(frame_bytes_action)
        latest_frame_action = frame_path_action

    return jsonify({"message": "Frame received successfully"}), 200


@app.route("/latest_frame")
def latest_frame_url():
    global latest_frame_action
    with lock:
        if latest_frame_action:
            return jsonify(
                {
                    "frame_url": url_for(
                        "static", filename=os.path.basename(latest_frame_action)
                    )
                }
            )
    return jsonify({"frame_url": None})


# here functionality end

# here start inspection demo functionality satrt


@app.route("/start_inspection")
def start_inspection():
    global inspection_started
    inspection_started = True  # Mark inspection as started

    try:
        with open("configs/config.json", "r") as f:
            status_data = json.load(f)

        # Deactivate steam mode for inspection

        status_data["status_inspection"] = True

        # Write back to the JSON file
        with open("configs/config.json", "w") as f:
            json.dump(status_data, f)

        return jsonify({"status": "inspection_started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/next_panel")
def next_panel():
    try:
        with open("configs/config.json", "r") as f:
            status_data = json.load(f)

        status_data["next_panel"] = True

        # Write back to the JSON file
        with open("configs/config.json", "w") as f:
            json.dump(status_data, f)

        return jsonify({"status": "next_panel_loaded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# here  functionality end


# here sattus of json function which control stream mode or detection mode
@app.route("/status")
def status():
    try:
        with open("configs/config.json", "r") as f:
            status_data = json.load(f)
            next_panel = status_data.get("next_panel", False)
            align_status = status_data.get("allign_status", False)
            barcode_scanned = status_data.get("is_barcode_scanned", False)

        return jsonify(
            {
                "allign_status": align_status,
                "is_barcode_scanned": barcode_scanned,
                "next_panel": next_panel,
            }
        )
    except FileNotFoundError:
        return jsonify({"error": "Status file not found"}), 500


def update_status(new_align_status, new_barcode_scanned_status):
    config_path = "configs/config.json"
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        config["allign_status"] = new_align_status
        config["is_barcode_scanned"] = new_barcode_scanned_status

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
    except FileNotFoundError:
        pass



# here  functionality end


# now defect detection functionality is satrt from here


@app.route("/save-image", methods=["POST"])
def save_image():
    try:
        data = request.get_json()

        image_data = data.get("imageData")
        defect_type = data.get("defectType")
        confidence = data.get("confidence")
        remarks = data.get("remarks")
        panel_number = data.get("panel")
        pcb_number = data.get("pcb")
        timestamp_str = data.get("timestamp", datetime.utcnow().isoformat())

        # Decode base64 image
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        # Create filename and save path
        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", ""))
        filename = f"defect_{timestamp.strftime('%Y%m%d%H%M%S%f')}.jpg"
        save_path = os.path.join(
            "static",
            "defect_images",
            secure_filename(filename),
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save image to disk
        with open(save_path, "wb") as f:
            f.write(image_bytes)

        # Create Defect record
        defect = Defect(
            timestamp=timestamp,
            image_path=save_path,
            defect_type=defect_type,
            confidence=confidence,
            remarks=remarks,
            panel_number=panel_number,
            pcb_number=pcb_number,
        )

        db.session.add(defect)
        db.session.commit()

        return jsonify({"message": "Defect saved successfully."}), 200

    except Exception as e:
        return jsonify({"message": "Error saving defect."}), 500


@app.route("/defect-history", methods=["GET"])
def defect_history():
    # Get pagination parameters
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 12))  # 12 cards per page
    
    query = Defect.query

    # Apply filters
    confidence_filter = request.args.get("confidence")
    if confidence_filter == "true":
        query = query.filter(Defect.confidence == 1.0)
    elif confidence_filter == "false":
        query = query.filter(Defect.confidence < 1.0)

    panel_filter = request.args.get("panel")
    if panel_filter:
        query = query.filter(Defect.panel_number.ilike(f"%{panel_filter}%"))

    pcb_filter = request.args.get("pcb")
    if pcb_filter:
        query = query.filter(Defect.pcb_number.ilike(f"%{pcb_filter}%"))

    defect_type_filter = request.args.get("defect_type")
    if defect_type_filter:
        query = query.filter(Defect.defect_type.ilike(f"%{defect_type_filter}%"))

    from_date = request.args.get("from")
    to_date = request.args.get("to")

    # Default to last 7 days if no date filter
    if not from_date and not to_date:
        today = datetime.now().date()
        week_ago = today - timedelta(days=7)
        query = query.filter(Defect.timestamp >= week_ago)

    if from_date:
        query = query.filter(Defect.timestamp >= from_date)
    if to_date:
        to_date_obj = datetime.strptime(to_date, "%Y-%m-%d")
        to_date_obj = to_date_obj + timedelta(days=1)  # Include the entire day
        query = query.filter(Defect.timestamp < to_date_obj)

    # Get total count for pagination
    total_count = query.count()
    
    # Apply pagination and ordering
    defects = query.order_by(Defect.timestamp.desc()).offset((page - 1) * per_page).limit(per_page).all()
    
    # Get summary statistics
    stats = {
        'total_defects': total_count,
        'high_confidence': query.filter(Defect.confidence >= 0.8).count(),
        'unique_panels': query.with_entities(Defect.panel_number).distinct().count(),
        'defect_types': query.with_entities(Defect.defect_type).distinct().count()
    }
    
    # Get available filter options
    available_panels = [p[0] for p in Defect.query.with_entities(Defect.panel_number).distinct().all() if p[0]]
    available_defect_types = [d[0] for d in Defect.query.with_entities(Defect.defect_type).distinct().all() if d[0]]
    
    pagination_info = {
        'page': page,
        'per_page': per_page,
        'total': total_count,
        'pages': (total_count + per_page - 1) // per_page,
        'has_prev': page > 1,
        'has_next': page < (total_count + per_page - 1) // per_page
    }
    
    return render_template("defect_history.html", 
                         defects=defects, 
                         stats=stats,
                         pagination=pagination_info,
                         available_panels=available_panels,
                         available_defect_types=available_defect_types)


# here functionality end


# now defect save in database from here
def save_to_json(user_id, defects, details):
    folder = "pcb_data"
    os.makedirs(folder, exist_ok=True)  # Ensure the directory exists

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Save defects.json file
    defects_file_name = f"{folder}/defects_{user_id}_{timestamp}.json"
    with open(defects_file_name, "w") as json_file:
        json.dump(defects, json_file, indent=4)

    # Save details.json file
    details_file_name = f"{folder}/details_{user_id}_{timestamp}.json"
    with open(details_file_name, "w") as json_file:
        json.dump(details, json_file, indent=4)

    return defects_file_name, details_file_name


@app.route("/save_pcb_data", methods=["POST"])
def save_pcb_data():
    try:
        # Import current_user from Flask-Login

        data = request.json
        defects = data.get("defects")
        details = data.get("details")

        # Use current_user.id instead of getting user_id from request
        user_id = current_user.id

        if not defects or not details:
            return jsonify({"error": "Missing defects or details"}), 400

        defects_file_path, details_file_path = save_to_json(user_id, defects, details)

        defect_entry = PCBDefect(
            user_id=user_id,  # Using current_user.id
            defect_data=defects_file_path,
            details_data=details_file_path,
            checklist_item=details.get("checklistItem"),
            points=details.get("points"),
            checked_items=details.get("checkedItems"),
            total_items=details.get("totalItems"),
            time_taken=details.get("timeTaken"),
            defect_found=details.get("defectFound"),
        )
        db.session.add(defect_entry)
        db.session.commit()

        return jsonify(
            {
                "message": "Data saved successfully",
                "defects_file": defects_file_path,
                "details_file": details_file_path,
            }
        ), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# here functionality end

# now analytic data save from here
from datetime import timedelta


@app.route("/analytics", methods=["GET"])
def get_analytic_data():
    try:
        start_date_str = request.args.get("start_date")
        end_date_str = request.args.get("end_date")
        user_id = request.args.get("user_id")
        pcb_type = request.args.get("pcb_type")
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 100))
        
        # Default to today's data if no date range specified
        # This ensures the analytics page shows only today's results on initial load
        if not start_date_str and not end_date_str:
            today = datetime.now().date()
            start_date_str = today.strftime("%Y-%m-%d")
            end_date_str = today.strftime("%Y-%m-%d")



        query = PCBDefect.query

        # Apply date filters
        if start_date_str:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            query = query.filter(PCBDefect.timestamp >= start_date)

        if end_date_str:
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            end_date = end_date + timedelta(days=1)
            query = query.filter(PCBDefect.timestamp < end_date)

        if user_id:
            query = query.filter(PCBDefect.user_id == user_id)
            
        if pcb_type:
            query = query.filter(PCBDefect.checklist_item == pcb_type)

        # Order by timestamp descending for latest first
        query = query.order_by(PCBDefect.timestamp.desc())
        
        # Get total count for pagination
        total_count = query.count()
        
        # Apply pagination to avoid loading millions of records at once
        defects = query.offset((page - 1) * per_page).limit(per_page).all()
        


        # Prepare lightweight response data - only load JSON files when specifically requested
        data = []
        for defect in defects:
            # Normalize defect_found: Defective = Fail, null/empty = Pass
            normalized_defect_found = "Pass"  # Default to Pass
            if defect.defect_found == "Defective":
                normalized_defect_found = "Fail"
            elif defect.defect_found == "Pass":
                normalized_defect_found = "Pass"
            # If null or empty, it remains "Pass"
            
            # Create the response object with basic data only
            defect_obj = {
                "id": defect.id,
                "user_id": defect.user_id,
                "timestamp": defect.timestamp.isoformat() if hasattr(defect.timestamp, "isoformat") else str(defect.timestamp),
                "checklist_item": defect.checklist_item or "Unknown",
                "points": defect.points or "0",
                "checked_items": defect.checked_items or 0,
                "total_items": defect.total_items or 0,
                "time_taken": defect.time_taken or "0",
                "defect_found": normalized_defect_found,
                "defect_file_path": defect.defect_data,
                "detail_file_path": defect.details_data,
                # Extract panel ID from checklist_item or use a default
                "panel_id": defect.checklist_item.split('_')[0] if defect.checklist_item and '_' in defect.checklist_item else defect.checklist_item or "Unknown"
            }
            data.append(defect_obj)

        return jsonify({
            "data": data,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total_count,
                "pages": (total_count + per_page - 1) // per_page
            }
        }), 200

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()

        return jsonify({"error": str(e), "traceback": error_details}), 500


@app.route("/analytics/summary", methods=["GET"])
def get_analytics_summary():
    """Get aggregated summary data for charts and stats - optimized for large datasets"""
    try:
        start_date_str = request.args.get("start_date")
        end_date_str = request.args.get("end_date")
        user_id = request.args.get("user_id")
        pcb_type = request.args.get("pcb_type")
        
        # Default to today's data if no date range specified  
        # This ensures the analytics summary shows only today's results on initial load
        if not start_date_str and not end_date_str:
            today = datetime.now().date()
            start_date_str = today.strftime("%Y-%m-%d")
            end_date_str = today.strftime("%Y-%m-%d")

        query = PCBDefect.query

        # Apply filters with error handling
        try:
            if start_date_str:
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                query = query.filter(PCBDefect.timestamp >= start_date)

            if end_date_str:
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                end_date = end_date + timedelta(days=1)
                query = query.filter(PCBDefect.timestamp < end_date)

            if user_id:
                query = query.filter(PCBDefect.user_id == user_id)
                
            if pcb_type:
                query = query.filter(PCBDefect.checklist_item == pcb_type)

        except ValueError as e:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

        # Use database aggregation for better performance
        from sqlalchemy import func, case
        
        # Get summary statistics
        # Logic: Defective = Fail, null/empty = Pass
        stats = db.session.query(
            func.count(PCBDefect.id).label('total_inspections'),
            func.count(case((PCBDefect.defect_found == 'Defective', 1))).label('fail_count'),
            func.count(case((PCBDefect.defect_found.is_(None), 1), (PCBDefect.defect_found == '', 1), (PCBDefect.defect_found == 'Pass', 1))).label('pass_count'),
            func.avg(func.cast(PCBDefect.time_taken, db.Float)).label('avg_time')
        ).filter(query.whereclause if query.whereclause is not None else True).first()
        
        # Get panel-wise data - use checklist_item directly as panel_id for simplicity
        # Logic: Defective = Fail, null/empty = Pass
        panel_stats = db.session.query(
            PCBDefect.checklist_item.label('panel_id'),
            func.count(PCBDefect.id).label('count'),
            func.count(case((PCBDefect.defect_found == 'Defective', 1))).label('fail_count'),
            func.count(case((PCBDefect.defect_found.is_(None), 1), (PCBDefect.defect_found == '', 1), (PCBDefect.defect_found == 'Pass', 1))).label('pass_count')
        ).filter(query.whereclause if query.whereclause is not None else True)\
         .group_by(PCBDefect.checklist_item)\
         .order_by(func.count(PCBDefect.id).desc())\
         .limit(20).all()  # Limit to top 20 panels
        
        # Get time-based data (hourly for today, daily for longer periods)
        # Logic: Defective = Fail, null/empty = Pass
        time_stats = []
        if start_date_str == end_date_str:  # Same day - hourly breakdown
            time_stats = db.session.query(
                func.strftime('%H', PCBDefect.timestamp).label('hour'),
                func.count(PCBDefect.id).label('count'),
                func.count(case((PCBDefect.defect_found == 'Defective', 1))).label('fail_count'),
                func.count(case((PCBDefect.defect_found.is_(None), 1), (PCBDefect.defect_found == '', 1), (PCBDefect.defect_found == 'Pass', 1))).label('pass_count')
            ).filter(query.whereclause if query.whereclause is not None else True)\
             .group_by(func.strftime('%H', PCBDefect.timestamp))\
             .order_by(func.strftime('%H', PCBDefect.timestamp)).all()
        else:  # Multiple days - daily breakdown
            time_stats = db.session.query(
                func.date(PCBDefect.timestamp).label('date'),
                func.count(PCBDefect.id).label('count'),
                func.count(case((PCBDefect.defect_found == 'Defective', 1))).label('fail_count'),
                func.count(case((PCBDefect.defect_found.is_(None), 1), (PCBDefect.defect_found == '', 1), (PCBDefect.defect_found == 'Pass', 1))).label('pass_count')
            ).filter(query.whereclause if query.whereclause is not None else True)\
             .group_by(func.date(PCBDefect.timestamp))\
             .order_by(func.date(PCBDefect.timestamp)).all()

        # Format response with safe calculations
        total_inspections = stats.total_inspections or 0
        pass_count = stats.pass_count or 0
        fail_count = stats.fail_count or 0
        
        summary = {
            "stats": {
                "total_inspections": total_inspections,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "pass_rate": round(pass_count / max(total_inspections, 1) * 100, 1) if total_inspections > 0 else 0,
                "fail_rate": round(fail_count / max(total_inspections, 1) * 100, 1) if total_inspections > 0 else 0,
                "avg_time": round(stats.avg_time or 0, 1)
            },
            "panel_data": [
                {
                    "panel_id": row.panel_id or "Unknown",
                    "total": row.count,
                    "pass": row.pass_count,
                    "fail": row.fail_count,
                    "pass_rate": round(row.pass_count / max(row.count, 1) * 100, 1)
                }
                for row in panel_stats
            ],
            "time_data": [
                {
                    "period": str(row[0]),  # hour or date
                    "total": row.count,
                    "pass": row.pass_count,
                    "fail": row.fail_count
                }
                for row in time_stats
            ]
        }
        
        return jsonify(summary), 200

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()

        return jsonify({"error": str(e), "traceback": error_details}), 500


@app.route("/analytics/defect-details/<int:defect_id>", methods=["GET"])
def get_defect_details(defect_id):
    """Get detailed defect data including JSON files - called on demand"""
    try:
        defect = PCBDefect.query.get_or_404(defect_id)
        
        defect_data = None
        detail_data = None

        # Read defect JSON file
        if defect.defect_data and os.path.exists(defect.defect_data):
            try:
                with open(defect.defect_data, "r") as f:
                    defect_data = json.load(f)
            except Exception as e:
                defect_data = {"error": f"Could not read file: {str(e)}"}

        # Read detail JSON file
        if defect.details_data and os.path.exists(defect.details_data):
            try:
                with open(defect.details_data, "r") as f:
                    detail_data = json.load(f)
            except Exception as e:
                detail_data = {"error": f"Could not read file: {str(e)}"}

        return jsonify({
            "id": defect.id,
            "defect_data": defect_data,
            "detail_data": detail_data
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# here functionality end

# here satrt templete setting functionality


def get_next_template_name():
    existing_templates = [f for f in os.listdir(TEMPLATES_PATH) if f.endswith(".json")]
    template_numbers = [
        int(f.split("_")[1].split(".")[0])
        for f in existing_templates
        if f.split("_")[1].split(".")[0].isdigit()
    ]
    next_number = max(template_numbers, default=0) + 1
    return f"template_{next_number}.json"


def load_template_data(template_id):
    template_file = os.path.join(TEMPLATES_PATH, f"{template_id}.json")
    if os.path.exists(template_file):
        with open(template_file, "r") as f:
            return json.load(f)
    return None


@app.route("/get_all_templates")
def get_all_templates():
    template_list = [
        f.split(".")[0] for f in os.listdir(TEMPLATES_PATH) if f.endswith(".json")
    ]
    return jsonify(template_list)


@app.route("/get_template_data/<template_id>", methods=["GET"])
def get_template_data(template_id):
    template_data = load_template_data(template_id)
    if template_data:
        return jsonify(template_data)
    return jsonify({"error": "Template not found"}), 404


@app.route("/save_template", methods=["POST"])
def save_template():
    data = request.get_json()
    template_id = data.get("template_id")

    if not template_id:
        template_id = get_next_template_name().split(".")[0]

    template_data = {
        "resistor": data.get("resistor", 0),
        "tolerance": data.get("tolerance", "0%"),
        "resistorInspection": data.get("resistorInspection", False),
        "capacitorColor": data.get("capacitorColor", "Brown"),
        "resistorColor": data.get("resistorColor", "Brown"),
        "capacitor": data.get("capacitor", 0),
        "port": data.get("port", 0),
        "image": data.get("image", "new_template_image.jpg"),
    }

    template_file = os.path.join(TEMPLATES_PATH, f"{template_id}.json")
    with open(template_file, "w") as f:
        json.dump(template_data, f, indent=4)

    return jsonify(
        {
            "message": f"Template {template_id} saved successfully!",
            "template_id": template_id,
        }
    )


@app.route("/static/images/<image_name>")
def serve_image(image_name):
    return send_from_directory(STATIC_IMAGES_PATH, image_name)


# here  functionality end

# here dummy isnpection api route


def load_config():
    """Load configuration file and return its content."""
    try:
        with open(
            "configs/config.json", "r"
        ) as config_file:
            return json.load(config_file)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def is_status_inspection_enabled():
    config = load_config()
    return config.get("status_inspection", False)


config_data = load_config()
serial_numbers = config_data.get("serial_no", [])

if isinstance(serial_numbers, str):
    serial_numbers = [serial_numbers]  # Convert string to list

panel_index = 0
pcb_index = 0
last_serial_numbers = serial_numbers


@app.route("/get_defects", methods=["GET"])
def get_defects():
    global panel_index, pcb_index, serial_numbers, last_serial_numbers

    if not is_status_inspection_enabled():
        return jsonify({"success": False, "message": "Inspection is disabled."}), 403

    time.sleep(1)

    config_data = load_config()
    serial_numbers = config_data.get("serial_no", [])

    if isinstance(serial_numbers, str):
        serial_numbers = [serial_numbers]

    if not serial_numbers:
        return jsonify(
            {"success": False, "message": "No serial numbers found in config."}
        ), 500

    if serial_numbers != last_serial_numbers:

        last_serial_numbers[:] = serial_numbers
        panel_index = 0
        pcb_index = 0

    current_panel = serial_numbers[panel_index]

    panel_path = f"results/{current_panel}/"
    available_pcbs = []

    if os.path.exists(panel_path) and os.path.isdir(panel_path):
        available_pcbs = [
            d
            for d in os.listdir(panel_path)
            if os.path.isdir(os.path.join(panel_path, d)) and d.startswith("PCB_")
        ]
        available_pcbs = sorted(available_pcbs)

    if not available_pcbs:
        return jsonify(
            {"success": False, "message": "No PCB folders available for this panel."}
        ), 404

    current_pcb = available_pcbs[pcb_index]
    pcb_index += 1

    if pcb_index >= len(available_pcbs):
        pcb_index = 0
        panel_index = (panel_index + 1) % len(serial_numbers)

    return jsonify({"Panel": current_panel, "PCB": current_pcb})


@app.route("/static/<path:filename>")
def serve_static_files(filename):
    return app.send_static_file(filename)


@app.route("/defect-image/<path:filename>")
def serve_defect_image(filename):
    """Serve defect images with proper error handling"""
    try:
        return send_from_directory("static/defect_images", filename)
    except FileNotFoundError:
        # Return a placeholder image or 404
        abort(404)


@app.route("/debug-defect-paths")
def debug_defect_paths():
    """Debug route to check defect image paths"""
    defects = Defect.query.limit(5).all()
    paths = []
    for d in defects:
        clean_path = d.image_path.replace('\\', '/') if d.image_path else None
        if clean_path and clean_path.startswith('static/'):
            filename = clean_path.replace('static/', '')
            image_url = url_for('static', filename=filename)
            file_exists = os.path.exists(clean_path)
        else:
            image_url = None
            file_exists = False
            
        paths.append({
            'id': d.id,
            'original_path': d.image_path,
            'clean_path': clean_path,
            'generated_url': image_url,
            'file_exists': file_exists
        })
    return jsonify(paths)


@app.route("/test-images")
def test_images():
    """Test page to verify defect images are working"""
    defects = Defect.query.limit(5).all()
    return render_template("test_images.html", defects=defects)


@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory("results", filename)


BASE_RESULTS_FOLDER = "results"


@app.route("/result/<path:panel_path>/pcb.png")
def serve_results_image(panel_path):
    folder_path = os.path.join(BASE_RESULTS_FOLDER, panel_path)
    file_path = os.path.join(folder_path, "pcb.png")

    if not os.path.exists(file_path):
        abort(404)
    return send_from_directory(folder_path, "pcb.png")


@app.route("/config_status")
def config_status():
    try:
        with open("configs/config.json", "r") as f:
            config = json.load(f)

        return jsonify(
            {
                "is_barcode_scanned": config.get("is_barcode_scanned", False),
                "template_pcbs_count": config.get("template_pcbs_count", 0),
                "image_paths": config.get("image_paths", []),
                "pcb_counter": config.get("pcb_counter", 0),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# end

if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    with app.app_context():
        db.create_all()
    app.run(host="127.0.0.1", port=5000, debug=True)
