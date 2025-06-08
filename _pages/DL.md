---
layout: default
title: DL
permalink: /DL/
parent: Category
---

<ul>
  {% assign cat = site.categories.DL %}
  {% for post in cat %}
    <li><a href="{{ post.url }}">{{ post.title }}</a> ({{ post.date | date: "%Y-%m-%d" }})</li>
  {% endfor %}
</ul>