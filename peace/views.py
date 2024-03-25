import hashlib
import uuid
from django.views import View
from .forms import AnswerForm,InterrogatorReportForm, DepSignUpForm, DepLoginForm
from .forms import SignUpForm, LoginForm 
from .models import Suspect, Case, SuspectResponse, Department, BadgeNumber
from .models import Enforcer, CaseCollection, EnforcerCase, SuspectCase, County



from django.http import HttpResponse
from django.shortcuts import render, redirect,get_object_or_404
from tabulate import tabulate
from django.conf import settings
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from io import BytesIO
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Spacer, Table, TableStyle, Paragraph



import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy as sp





def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            enforcer = form.save(commit=False)
            enforcer.set_password(form.cleaned_data['password'])
            enforcer.save()
            return redirect('login')
    else:
        form = SignUpForm()
    return render(request, 'signup.html', {'form': form})


def depsignup(request):
    if request.method == 'POST':
        form = DepSignUpForm(request.POST)
        if form.is_valid():
            department = form.save(commit=False)
            department.set_password(form.cleaned_data['password'])
            department.save()
            return redirect('deplogin')
    else:
        form = DepSignUpForm()
    return render(request, 'depsignup.html', {'form': form})


class LoginView(View):
    def get(self, request):
        form = LoginForm()
        return render(request, 'login.html', {'form': form})

    def post(self, request):
        form = LoginForm(request.POST)
        if form.is_valid():
            badge_no = form.cleaned_data['badge_no']
            password = form.cleaned_data['password']
            enforcer = Enforcer.objects.filter(badge_no=badge_no).first()
            if enforcer and enforcer.check_password(password):
                # Authentication successful
                #request.session['badge_no'] = badge_no
                return redirect('forms')  # Redirect to the dashboard upon successful login
            else:
                # Authentication failed
                form.add_error(None, 'Invalid badge number or password')
        return render(request, 'login.html', {'form': form})

class DepLoginView(View):
    def get(self, request):
        form = DepLoginForm()
        return render(request, 'deplogin.html', {'form': form})

    def post(self, request):
        form = DepLoginForm(request.POST)
        if form.is_valid():
            dep_no = form.cleaned_data['dep_no']
            password = form.cleaned_data['password']
            department = Department.objects.filter(dep_no=dep_no).first()
            if department and department.check_password(password):
                # Authentication successful
                return redirect('index')  # Redirect to the home page upon successful login
            else:
                # Authentication failed
                form.add_error(None, 'Invalid Department number or password')
        return render(request, 'deplogin.html', {'form': form})




class ErrorPageView(View):
    def get(self,request):
        return render(request, 'interrogator_error.html')  
      
class SuccessPageView(View):
    def get(self, request, *args, **kwargs):
        serial_number = self.kwargs.get('serial_number', None)
        return render(request, 'interrogator_success.html', {'serial_number': serial_number})

class HomePageView(View):
    def get(self,request):
        return render(request, 'index.html')

class InterrogatorDashboardView(View):
    def get(self,request):
        return render(request, 'interrogator_dashboard.html')
    
    
class DashboardView(View):
    def get(self,request):
        return render(request, 'dashboards.html')
    
    
# views.py
from django.shortcuts import render, redirect
from django.contrib.auth import logout as django_logout

def logout(request):
    if request.method == 'POST':
        django_logout(request)
        return redirect('home')  # Redirect to home page after logout
    return render(request, 'logout.html')




from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.conf import settings
from django.utils.crypto import get_random_string
from .forms import PasswordResetForm
from .models import PasswordResetToken

def reset_password(request):
    if request.method == 'POST':
        form = PasswordResetForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            user = User.objects.filter(email=email).first()
            if user:
                # Generate a unique token
                token = get_random_string(length=32)
                # Save the token to the database
                PasswordResetToken.objects.create(user=user, token=token)
                # Send password reset email
                reset_link = request.build_absolute_uri('/') + f'reset-password/{token}'
                send_mail(
                    'Reset Your Password',
                    f'Click the link to reset your password: {reset_link}',
                    settings.EMAIL_HOST_USER,
                    [email],
                    fail_silently=False,
                )
            else:
                # Display error message for non-existing email
                error_message = "Email does not exist in our records."
                return render(request, 'reset_password.html', {'form': form, 'error_message': error_message})
            # Redirect to a success page or show a message
            return redirect('password_reset_done')
    else:
        form = PasswordResetForm()
    return render(request, 'reset_password.html', {'form': form})

def reset_password_confirm(request, token):
    # Check if the token exists in the database
    password_reset_token = PasswordResetToken.objects.filter(token=token).first()
    if password_reset_token:
        if request.method == 'POST':
            # Update the user's password
            user = password_reset_token.user
            new_password = request.POST.get('new_password')
            user.set_password(new_password)
            user.save()
            # Delete the used token
            password_reset_token.delete()
            # Redirect to password reset success page or login page
            return redirect('password_reset_complete')
        else:
            # Render a form for the user to enter new password
            return render(request, 'reset_password_confirm.html', {'token': token})
    else:
        # Token is invalid or expired, handle appropriately
        return render(request, 'reset_password_token_invalid.html')



def generate_serial_number(unique_id, case_description):
    data_string = f"{unique_id}-{case_description}"
    unique_identifier = str(uuid.uuid4())
    combined_string = f"{data_string}-{unique_identifier}"
    serial_number = hashlib.md5(combined_string.encode()).hexdigest()
    return serial_number

class FormsView(View):
    template_name = 'forms.html'
    
    def get(self, request):
        answer_form = AnswerForm()
        report_form = InterrogatorReportForm()
        return render(request, self.template_name, {'answer_form': answer_form, 'report_form': report_form})

    def post(self, request):
        answer_form = AnswerForm(request.POST)
        report_form = InterrogatorReportForm(request.POST)

        if answer_form.is_valid():
            case_description = answer_form.cleaned_data['case_description']
            unique_id = answer_form.cleaned_data['unique_id']
            trace = answer_form.cleaned_data['trace']
            know_complainant = answer_form.cleaned_data['know_complainant']
            involved_with_complainant = answer_form.cleaned_data['involved_with_complainant']
            recidivist = answer_form.cleaned_data['recidivist']
            question1 = answer_form.cleaned_data['question1']
            question2 = answer_form.cleaned_data['question2']
            question3 = answer_form.cleaned_data['question3']
            query1 = answer_form.cleaned_data['query1']
            query2 = answer_form.cleaned_data['query2']
            query3 = answer_form.cleaned_data['query3']
            
            
            # Check for existing testification
            if SuspectResponse.objects.filter(case_description=case_description, unique_id=unique_id).exists():
                form.add_error(None, "Intorragation Information for This Suspect ID and Case Description already submitted, sorry.")
                return render(request, self.template_name, {'answer_form': form})

            
            serial_number = generate_serial_number(unique_id, case_description)
            suspectResponse = answer_form.save(commit=False)
            suspectResponse.serial_number = serial_number
            suspectResponse.save()

            return redirect('success', serial_number=serial_number)
        
        elif report_form.is_valid():
            serial_number = report_form.cleaned_data['serial_number']
            try:
                suspect_response = get_object_or_404(SuspectResponse, serial_number=serial_number)
                suspect = suspect_response.unique_id
                if not SuspectResponse.objects.filter(serial_number=serial_number).exists():
                    form.add_error(None, "Case Information for the Provided Serial Number does not exist, sorry.")
                    return render(request, self.template_name, {'report_form': form})                
            except SuspectResponse.DoesNotExist:
                return render(request, 'interrogator_error.html', {'error_message': f'Suspect Report with serial number "{serial_number}" not found.'})
        
        
        
            # My processing for generating report data
            mlm = MachineLearningModel()
            accuracy = mlm.accuracy()
            sent1 = SentimentAnalyser()
            criminal = CriminalPrediction()
            name = suspect.unique_id
            age = suspect.age
            gender = suspect.gender
            recidivist = suspect_response.recidivist
            firstResponse = suspect_response.trace
            secondResponse = suspect_response.know_complainant
            question1 = suspect_response.question1
            question2 = suspect_response.question2
            question3 = suspect_response.question3
            query1 = suspect_response.query1
            query2 = suspect_response.query2
            query3 = suspect_response.query3
            consistency_score = sent1.calculate_consistency_score(question1, query1)
            trace = suspect_response.trace
            honesty_score = sent1.is_honest(question2)
            criminal.data_retrieval(name, age, recidivist, trace, obedient_score, consistency_score, gender)
            criminal.data_preparation()
            result = criminal.result()
            
            sentiment_analyzer = SentimentAnalyzer()
            text_groups = {
                'group1': ("Text1-1", "Text1-2"),
                'group2': ("Text2-1", "Text2-2"),
                'group3': ("Text3-1", "Text3-2")
            }
            
            consistency_score = sentiment_analyzer.calculate_consistency_score(text_groups)

            
            
            report_data = {
                'name': name,
                'age':age,
                'gender':gender,
                'recidivist':recidivist,
                'honesty_score':honesty_score,
                'consistency_score':consistency_score,
                'case_description': suspect_response.case_description,
                'result': result,
                'accuracy': accuracy,
            }
            
            
            # Generate PDF
            pdf_bytes = generate_pdf(report_data)
               
            # Return the PDF file as a response
            response = HttpResponse(pdf_bytes, content_type='interrogator/pdf')
            response['Content-Disposition'] = f'attachment; filename="{name}_report.pdf"'
            return response
            
            
            # If neither form is valid or processed, return the template with both forms       
        return render(request, self.template_name, {'answer_form': answer_form, 'report_form': report_form})



import nltk
import spacy

class SentimentAnalyzer:
    def __init__(self):
        """
        Initialize SentimentAnalyzer with NLTK Sentiment Intensity Analyzer and spaCy NLP model.
        """
        self.sid = nltk.sentiment.SentimentIntensityAnalyzer()
        self.nlp = spacy.load("en_core_web_sm")

    def is_honest(self, text: str) -> str:
        """
        Determine honesty based on sentiment score.
        
        Parameters:
            text (str): Input text to analyze.
        
        Returns:
            str: 'Yes' if sentiment score is positive or neutral, 'No' otherwise.
        """
        sentiment_scores = self.sid.polarity_scores(text)
        honesty_score = sentiment_scores['compound']
        return 'Yes' if honesty_score >= 0 else 'No'

    def calculate_consistency_score(self, text_groups: dict) -> int:
        """
        Calculate consistency score based on multiple pairs of texts.
        
        Parameters:
            text_groups (dict): Dictionary where the keys represent group names
                and the values are tuples containing two texts.
        
        Returns:
            int: Average consistency score rounded to the nearest integer.
        """
        consistency_scores = []
        for key, (text1, text2) in text_groups.items():
            try:
                doc1 = self.nlp(text1)
                doc2 = self.nlp(text2)
                similarity = doc1.similarity(doc2)
                consistency_scores.append(similarity)
            except Exception as e:
                # Return a default value if an error occurs during consistency score calculation
                return 0
        
        average_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
        return round(average_consistency * 100)



class MachineLearningModel:
    def __init__(self):
        self.new_data = None
        self.prediction = None
        self.numeric_feature = None
        self.categorical_feature = None
        self.training_features = None
        self.predictions = None
        self.model = None
        self.scaler = None
        self.actual_labels = None
        self.file_path = os.path.join(settings.BASE_DIR, 'peace', 'peace', 'crime.csv')
        self.df = pd.read_csv(self.file_path)
        self.feature_names = ['Age','Recidivist','Trace','Honest','ConsistencyScore','Gender']
        self.training_features = self.df[self.feature_names]
        self.outcome_name = ['Criminal']
        self.outcome_labels = self.df[self.outcome_name]
        self.numeric_feature_names = ['Age','ConsistencyScore']
        self.categorical_feature_names = ['Recidivist','Trace','Honest','Gender']
        ss = StandardScaler()
        ss.fit(self.training_features[self.numeric_feature_names])
        self.training_features[self.numeric_feature_names] = ss.transform(self.training_features[self.numeric_feature_names])
        self.training_features = pd.get_dummies(self.training_features,columns=self.categorical_feature_names)
        self.categorical_engineered_features = list(set(self.training_features.columns)-set(self.numeric_feature_names))
        self.lr = LogisticRegression()
        self.model = self.lr.fit(self.training_features,np.array(self.outcome_labels['Criminal']))
        self.pred_labels = self.model.predict(self.training_features)
        self.actual_labels = np.array(self.outcome_labels['Criminal'])
        if not os.path.exists('Model'):
            os.mkdir('Model')
        if not os.path.exists('Scaler'):
            os.mkdir('Scaler')
        joblib.dump(self.model,r'Model/model.pickle')
        joblib.dump(ss,r'Scaler/scaler.pickle')
        self.model = joblib.load(r'Model/model.pickle')
        self.scaler = joblib.load(r'Scaler/scaler.pickle')

    def accuracy(self):
        acc = accuracy_score(self.actual_labels, self.pred_labels)
        class_stats = classification_report(self.actual_labels, self.pred_labels)

        accuracy_row = f"{acc * 100:.2f}%"
        class_stats_row = "Classification Stats:\n" + class_stats

        return accuracy_row

class CriminalPrediction:
    def __init__(self):
        self.new_data = None
        self.prediction = None
        self.training_features = None
        self.numeric_feature = None
        self.categorical_feature = None
        self.model = joblib.load(r'Model/model.pickle')
        self.scaler = joblib.load(r'Scaler/scaler.pickle')
        self.prediction = None
        self.predictions = None
        self.file_path = os.path.join(settings.BASE_DIR, 'peace', 'peace', 'crime.csv')
        self.df = pd.read_csv(self.file_path)
        self.feature_names = ['Age','Recidivist','Trace','Honest','ConsistencyScore','Gender']
        self.training_features = self.df[self.feature_names]
        self.outcome_name = ['Criminal']
        self.outcome_labels =self.df[self.outcome_name]
        self.numeric_feature_names = ['Age','ConsistencyScore']
        self.categorical_feature_names = ['Recidivist', 'Trace','Honest','Gender']
        self.ss = StandardScaler()
        self.ss.fit(self.training_features[self.numeric_feature_names])
        self.training_features[self.numeric_feature_names] = self.ss.transform(self.training_features[self.numeric_feature_names])
        self.training_features = pd.get_dummies(self.training_features,columns=self.categorical_feature_names)
        
    
    def data_retrieval(self,name,age,recidivist,trace,honest,consistency_score,gender):
        self.new_data = pd.DataFrame([{'Name': name,
                        'Age': age,
                        'Recidivist': recidivist,
                        'Trace': trace,
                        'Honest': honest,
                        'ConsistencyScore': consistency_score,
                        'Gender': gender
                        }])
    
    def data_preparation(self):
        self.prediction = self.new_data.copy()
        self.numeric_feature = ['Age','ConsistencyScore']
        self.categorical_feature = ['Recidivist_Yes','Recidivist_No','Trace_Yes','Trace_No','Honest_Yes','Honest_No','Gender_Male','Gender_Female']
        self.prediction[self.numeric_feature] = self.scaler.transform(self.prediction[self.numeric_feature])
        self.prediction= pd.get_dummies(self.prediction,columns=['Recidivist', 'Trace','Honest','Gender'])
        for feature in self.categorical_feature:
            if feature not in self.prediction.columns:
                self.prediction[feature] = 0 #Add missing categorical feature columns with 0 columns
        self.prediction = self.prediction[self.training_features.columns]
        self.predictions = self.model.predict(self.prediction)
        self.new_data['Criminal'] = self.predictions

    def result(self):
        result_str = ""
        for index, row in self.new_data.iterrows():
            result_str += f"{'Yes' if row['Criminal'] == 1 else 'No'}"
        return result_str
       


def generate_pdf(report_data):
    buffer = BytesIO()
    pdf_canvas = SimpleDocTemplate(buffer, pagesize=landscape(letter))

    # Set font and size
    styles = getSampleStyleSheet()
    style_normal = styles['Normal']
    style_heading = styles['Heading1']
    style_normal.fontName = 'Helvetica'
    style_heading.fontName = 'Helvetica'
    style_normal.fontSize = 12
    style_heading.fontSize = 14

    # Define additional information to include at the top
    additional_info = [
        "PEACE 2017 REFORMED, THE PEACE DIGITAL",
        " ",
        "P.O. BOX. 197-40400 SARE-AWENDO",
        " ",
        "Email: bornfacesonye@gmail.com",
        
        "TEL: +254-798073204",
        " ",
        " ",
        "\n",
        " ",
        "DEPARTMENT:KAMITI",
        " ",
        " ",
        " ",
        " ",
        "CASE YEAR: 2024",
        " \n\n",
    ]

    # Create a paragraph for additional information
    additional_info_paragraphs = [Paragraph(info, style_normal) for info in additional_info]

    # Calculate the height of the additional information
    additional_info_height = sum(paragraph.wrapOn(pdf_canvas, pdf_canvas.width, pdf_canvas.height)[1] for paragraph in additional_info_paragraphs)

    # Define table data
    table_data = [
        ["Case Information"],
        ["Case Description:", report_data['case_description']],
        ["Suspect Unique Identification:", report_data['name']],
        ["Suspect Age:", report_data['age']],
        ["Suspect Gender:", report_data['gender']],
        ["Have The Suspect been in Similar Case Before:", report_data['recidivist']],
        ["Is The Suspect Honest:", report_data['honesty_score']],
        ["Suspect Consistency Score:", report_data['consistency_score']],
        ["Predicted Criminal:", report_data['result']],
        ["Accuracy of the Model Used:", report_data['accuracy']],
        
    ]

    # Calculate the required height for the table
    table_height = len(table_data) * style_normal.fontSize

    # Adjust the page height based on the content
    pdf_canvas.pagesize = landscape(letter)
    pdf_canvas.height = max(additional_info_height, table_height) + inch  # Add some extra space between content and table

    # Create a table for main data
    main_table = Table(table_data, colWidths=[6*inch, 4*inch])  # Adjusting column widths

    # Set table style for main data
    style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),  # Header background color
                        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),  # Content background color
                        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 0), (-1, -1), 12),
                        ('LEFTPADDING', (0, 0), (-1, -1), 10),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                        ('TOPPADDING', (0, 0), (-1, -1), 5),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('SPAN', (0, 0), (1, 0)),  # Merge cells for the heading
                        ])

    main_table.setStyle(style)

    # Add page border
    frame_styling = TableStyle([('BOX', (0, 0), (-1, -1), 2, colors.black)])
    main_table.setStyle(frame_styling)

    # Add tables to the PDF
    elements = additional_info_paragraphs + [Spacer(1, 0.25 * inch), main_table]

    # Build PDF
    pdf_canvas.build(elements)

    # Save the PDF file
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes