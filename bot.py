from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Updater, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler
from pydub import AudioSegment
import os
from model import PretrainedResNet  # Импорт класса модели
import librosa
import numpy as np
import torch
from transforms import ToMelSpectrogram  # Импорт функции для преобразования в мел-спектрограмму
import time 
TOKEN = '7033502693:AAGuv08AlR_ez3Oo-jqq3AQsBTwUNptRVpE'
result = []
model = PretrainedResNet(num_classes=7)  # Инициализация модели
model.load_state_dict(torch.load('models/trained_model (2).pth', map_location=torch.device('cpu')))  # Загрузка весов модели
model.eval()

label_to_index = {0: 'Здоровый', 1: 'Сердечная недостаточность', 2: 'Бронхит', 3: 'ОРВИ', 4: 'Пневмония', 5: 'ХОБЛ', 6: 'Астма'}  # Маппинга меток


def start(update: Update, context: CallbackContext):
    welcome_message = (
        "Добро пожаловать! Этот проект собирает датасет дыханий людей с различными заболеваниями дыхательных путей.\n"
        "Пожалуйста, запишите аудио своего дыхания.\n"
        "Для корректного определения заболевания рекомендуется записывать аудио длинною от 15 до 20 секунд и в тишине!"
    )
    update.message.reply_text(welcome_message)


def handle_voice(update: Update, context: CallbackContext):
    file = update.message.voice.get_file()
    file.download('user_voice.ogg')

    audio = AudioSegment.from_ogg('user_voice.ogg')
    audio.export('user_voice.wav', format='wav')

    # Отправить аудио на предсказание
    result.append(predict('user_voice.wav'))
    update.message.reply_text(f"Предсказание: {label_to_index[result[0]]}")

    # Предложить пользователю подтвердить диагноз
    keyboard = [
        [InlineKeyboardButton("Правильный диагноз", callback_data='correct')],
        [InlineKeyboardButton("Неправильный диагноз", callback_data='incorrect')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Правильный ли это диагноз?', reply_markup=reply_markup)

def button(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()

    if query.data == 'correct':
        save_correct_data('user_voice.wav', result[0])
        result.clear()
        query.edit_message_text(text="Спасибо! Ваши данные сохранены.")
    elif query.data == 'incorrect':
        keyboard = [
            [InlineKeyboardButton("Здоровый", callback_data='0')],
            [InlineKeyboardButton("Сердечная недостаточность ", callback_data='1')],
            [InlineKeyboardButton("Бронхит", callback_data='2')],
            [InlineKeyboardButton("ХОБЛ", callback_data='5')],
            [InlineKeyboardButton("Астма", callback_data='6')],
            [InlineKeyboardButton("Пневмония", callback_data='4')],
            [InlineKeyboardButton("ОРВИ", callback_data='3')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        query.edit_message_text(text="Пожалуйста, выберите правильный диагноз:", reply_markup=reply_markup)
    else:
        save_incorrect_data('user_voice.wav', query.data)
        query.edit_message_text(text="Спасибо за исправление! Ваши данные сохранены.")

def predict(audio_data):
    signal, sr = librosa.load(audio_data, sr=None)
    mel_spectrogram = ToMelSpectrogram()(signal)
    
    input_tensor = torch.tensor(mel_spectrogram).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()


   

def save_correct_data(audio_path, correct_label):
    current_time_millis = int(time.time() * 1000)
    if not os.path.exists('correct'):
        os.makedirs('correct')
    os.rename(audio_path, f"correct/{correct_label}_{current_time_millis}_{os.path.basename(audio_path)}")

def save_incorrect_data(audio_path, correct_label):
    current_time_millis = int(time.time() * 1000)
    if not os.path.exists('incorrect'):
        os.makedirs('incorrect')
    os.rename(audio_path, f"incorrect/{correct_label}_{current_time_millis}_{os.path.basename(audio_path)}")

def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(filters.Filters.voice, handle_voice))
    dp.add_handler(CallbackQueryHandler(button))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()